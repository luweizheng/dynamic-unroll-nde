import sys
import math
import time
import functools
import argparse
from typing import Sequence, Callable


import numpy as np
import xgboost as xgb

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import optax  # https://github.com/deepmind/optax


sys.path.insert(0, '..')
from simulated_annealing import annealing


class FNN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size, out_size, width_size=width_size, depth=depth, key=key
        )

    def __call__(self, t, x, train: bool = True):
        y, f_vjp = jax.vjp(lambda x: self.mlp(jnp.concatenate([t, x])), x)
        dudx = f_vjp(jnp.ones(y.shape))
        return y, dudx[0]


class FBSDEStep(eqx.Module):
    unet: FNN
    noise_size: int
    mu_fn: Callable = eqx.static_field()
    sigma_fn: Callable = eqx.static_field()
    phi_fn: Callable = eqx.static_field()

    def __init__(self, in_size, out_size, width_size, depth, noise_size, key):
        self.unet = FNN(in_size=in_size, out_size=out_size,
                        width_size=width_size, depth=depth, key=key)

        def mu_fn(t, X, Y, Z):
            del t, Y, Z
            return jnp.zeros_like(X)

        def sigma_fn(t, X, Y):
            del t, Y
            return 0.4 * X

        def phi_fn(t, X, Y, Z):
            del t
            return 0.05 * (Y - jnp.sum(X * Z, keepdims=True))

        self.mu_fn = mu_fn
        self.sigma_fn = sigma_fn
        self.phi_fn = phi_fn
        self.noise_size = noise_size

    def u_and_dudx(self, t, x):
        return self.unet(t, x)

    def __call__(self, carry, inp):
        # `t` and `W` are (batch_size, num_timestep, dim)
        # it have input data across iterations

        (i, t0, dt, x0, y0, z0, key) = carry

        key, key2 = jrandom.split(key)

        # use `i` to index input data
        curr_t = jnp.full((1, ), t0 + i * dt)
        next_t = jnp.full((1, ), t0 + (i + 1) * dt)

        dW = jrandom.normal(key, (self.noise_size, )) * jnp.sqrt(dt)

        x1 = x0 + self.mu_fn(curr_t, x0, y0, z0) * dt + \
            self.sigma_fn(curr_t, x0, y0) * dW

        y1_tilde = y0 + self.phi_fn(curr_t, x0, y0, z0) * dt + \
            jnp.sum(z0 * self.sigma_fn(curr_t, x0, y0) * dW, keepdims=True)

        y1, z1 = self.unet(next_t, x1)

        carry = (i+1, t0, dt, x1, y1, z1, key2)
        outputs = (x1, y1_tilde, y1)
        return carry, outputs


class NeuralFBSDE(eqx.Module):
    step: FBSDEStep
    hidden_size: int
    depth: int
    width_size: int
    unroll : int
    num_timesteps: int

    def __init__(self, in_size, out_size, width_size, depth, noise_size, num_timesteps , unroll, key):
        self.step = FBSDEStep(
            in_size, out_size, width_size, depth, noise_size, key)
        self.hidden_size = in_size - 1
        self.depth = depth
        self.width_size = width_size
        self.unroll =unroll
        self.num_timesteps = num_timesteps

    def make_cost_model_feature(self):

        def step_fn(carry, inp):
            return self.step(carry, inp)

        dummy_t0 = 0.0
        dummy_dt = 0.2

        x0 = jnp.ones((self.hidden_size, ))
        y0, z0 = self.step.u_and_dudx(t=jnp.zeros((1, )), x=x0)

        dummy_bm_key = jrandom.PRNGKey(0)
        carry = (0, dummy_t0, dummy_dt, x0, y0, z0, dummy_bm_key)

        hlo_module = jax.xla_computation(step_fn)(carry, None).as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        step_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(
            client, hlo_module)
        step_bytes_access = step_cost['bytes accessed']
        step_bytes_access_op0 = step_cost['bytes accessed operand 0 {}']
        step_bytes_access_op1 = step_cost['bytes accessed operand 1 {}']
        step_bytes_access_out = step_cost['bytes accessed output {}']
        step_flops = step_cost['flops']

        features = []
        # step bytes access
        features.append(step_bytes_access)
        features.append(step_bytes_access_op0)
        features.append(step_bytes_access_op1)
        features.append(step_bytes_access_out)

        # step FLOPS
        features.append(step_flops)
        # step Arithmetic Intensity
        features.append(step_flops / step_bytes_access)

        total_params = sum(p.size for p in jtu.tree_leaves(
            eqx.filter(self.step, eqx.is_array)))

        # total params
        features.append(total_params / 1e6)

        # hidden_size: the dimension of DE
        features.append(self.hidden_size)


        # depth: depth of all Dense layers
        features.append(self.depth)
        # depth of width <= 128
        features.append(self.depth)
        # depth of width <= 256
        features.append(0)
        # depth of width <= 512
        features.append(0)
        # depth of width > 512
        features.append(0)

        return features

    def __call__(self, t0, dt, x0):
        
        key=jrandom.PRNGKey(0)

        y0, z0 = self.step.u_and_dudx(t=jnp.zeros((1, )), x=x0)

        carry = (0, t0, dt, x0, y0, z0, key)
        

        def step_fn(carry, inp=None):
            return self.step(carry, inp)

        (carry, output) = jax.lax.scan(step_fn, carry,
                                       None, length=self.num_timesteps, unroll=self.unroll)
        return (carry, output)


@jax.jit
def sum_square_error(y, y_pred):
    """Computes the sum of square error."""
    return jnp.sum(jnp.square(y - y_pred))


def g_fn(X):
    return jnp.sum(X ** 2, axis=-1, keepdims=True)


def dg_fn(X):
    y, vjp_func = jax.vjp(g_fn, X)
    return vjp_func(jnp.ones(y.shape))[0]


@eqx.filter_jit
def train_step(model, x0, t0, dt, num_timesteps, optimizer, opt_state, unroll=1, key=jrandom.PRNGKey(0)):
    # batch_size = model.batch_size
    # t, W = data
    @eqx.filter_jit
    def loss_fn(model):
        loss = 0.0
        fn = functools.partial(model, t0, dt, num_timesteps, unroll)
        # out_carry, out_val = jax.vmap(model, in_axes=(0, None, None, None, None, 0))(x0, t0, dt, num_timesteps, unroll, key)
        out_carry, out_val = jax.vmap(fn, in_axes=(0, 0))(x0, key)
        (_, _, _, x_final, y_final, z_final, _) = out_carry
        (x, y_tilde_list, y_list) = out_val

        loss += sum_square_error(y_tilde_list, y_list)
        loss += sum_square_error(y_final, g_fn(x_final))
        loss += sum_square_error(z_final, dg_fn(x_final))

        return (loss, y_list)

    (loss, y), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state, y


def predict_unroll(args):
    learning_rate = args.lr
    rng = jrandom.PRNGKey(args.seed)

    model = NeuralFBSDE(in_size=args.dim + 1, out_size=1,
                        width_size=args.width_size, depth=args.depth, noise_size=args.dim, 
                        num_timesteps=args.num_timesteps, unroll=args.unroll, key=rng)
    
    features = model.make_cost_model_feature()
    features.append(args.batch_size)
    features.append(args.num_timesteps)

    compile_model_loaded = xgb.Booster()
    compile_model_loaded.load_model(args.xgb_dir+"titan_compile.txt")

    run_model_loaded = xgb.Booster()
    run_model_loaded.load_model(args.xgb_dir+"titan_execution.txt")
    
    predict_list=[]
    
    def cost_fn(unroll):
        cur_features = features + [unroll]
        cur_features = np.array(cur_features, dtype=object)
        
        compilation_time_pred = compile_model_loaded.predict(xgb.DMatrix([cur_features]))
        run_time_pred = run_model_loaded.predict(xgb.DMatrix([cur_features]))
        total_time_pred = compilation_time_pred + run_time_pred * 50 # suppose 50000 iters then x/1000 * 50000/1000
        
        return total_time_pred
    
    # exhaustively iterate a list of candidates
    unroll_list = [1, 2, 5, 10, 20, 25, 50, 100]
    total_time_pred_list = []
    for unroll in unroll_list:
        total_time_pred = cost_fn(unroll)
        total_time_pred_list.append(total_time_pred)
        
    predicted_unroll = unroll_list[np.argmin(total_time_pred_list)]
    predict_list.append(predicted_unroll)
    
    # scipy sa
    bounds = [[2, args.num_timesteps//2]]
    from scipy.optimize import dual_annealing

    result = dual_annealing(cost_fn, bounds, maxiter=args.max_steps)
    predicted_unroll = result['x'][0]
    predict_list.append(predicted_unroll)
    
    # my own implementation of SA
    bounds = (2, args.num_timesteps//2)
    def clip(x, bounds):
        """ Force x to be in the interval."""
        a, b = bounds
        return int(max(min(x, b), a))

    def random_neighbour(x, bounds, fraction=1):
        """Move a little bit x, from the left or the right."""
        amplitude = (max(bounds) - min(bounds)) * fraction / 10
        delta = (-amplitude/2.) + amplitude * np.random.random_sample()
        return clip(x + delta, bounds)
    predicted_unroll, _, _, _ = annealing(bounds, cost_fn, random_neighbour=random_neighbour, maxsteps=args.max_steps, debug=False)
    predict_list.append(predicted_unroll)
    
    # print result
    print("exhaustive, sa_scipy, sa_our")
    print(','.join(map(str, predict_list)))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width-size', type=int, default=64)
    parser.add_argument('--num-timesteps', type=int, default=100)
    parser.add_argument('--num-iters', type=int, default=1000)
    parser.add_argument('--unroll', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=5678)
    parser.add_argument('--xgb-dir', type=str, default='../cost-model/ckpt/')

    args = parser.parse_args()

    predict_unroll(args)
    

if __name__ == '__main__':
    main()
