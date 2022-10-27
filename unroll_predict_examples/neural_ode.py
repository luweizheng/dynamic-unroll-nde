import time
import functools
import argparse

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

import diffrax # https://github.com/patrick-kidger/diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import optax  # https://github.com/deepmind/optax

import sys; 
sys.path.insert(0, '..')
from simulated_annealing import annealing

_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args=None):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func
    hidden_size:int
    width_size:int
    depth:int

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = data_size
        self.width_size = width_size
        self.depth = depth
        self.func = Func(data_size, width_size, depth, key=key)
    # https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Ralston's_method
    def ralston_step_fn(self, carry):
        (i, t0, dt, y0) = carry
        t1 = t0 + dt
        k1 = self.func(t0, y0, args=None)
        k2 = self.func(t0 + 0.5 * dt, y0 + 0.5 * k1, args=None)
        k3 = self.func(t0 + 3/4 * dt, y0 + 3/4 * k2)
        y1 = (2 / 9 * k1 + 1 / 3 * k2 + 4 / 9 * k3) * dt + y0
        carry = (i+1, t1, dt, y1)
        return (carry , y1)
    
    # https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Classic_fourth-order_method
    def rk4_step_fn(self, carry):
        (i, t0, dt, y0) = carry
        t1 = t0 + dt
        half_dt = dt * 0.5
        k1 = self.func(t0, y0, args=None)
        k2 = self.func(t0 + half_dt, y0 + half_dt * k1, args=None)
        k3 = self.func(t0 + half_dt, y0 + half_dt * k2)
        k4 = self.func(t1, y0 + dt * k3, args=None)
        y1 = (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth + y0
        carry = (i+1, t1, dt, y1)
        return (carry , y1)

    def rk4_alt_step_fn(self, carry):
        (i, t0, dt, y0) = carry
        t1 = t0 + dt
        k1 = self.func(t0, y0, args=None)
        k2 = self.func(t0 + dt * _one_third, y0 + dt * k1 * _one_third, args=None)
        k3 = self.func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
        k4 = self.func(t1, y0 + dt * (k1 - k2 + k3), args=None)
        y1 = (k1 + 3 * (k2 + k3) + k4) * dt * 0.125 + y0
        carry = (i+1, t1, dt, y1)
        return (carry , y1)
    
    def euler_step_fn(self, carry):
        (i, t0, dt, y0) = carry
        t1 = t0 + dt
        dy = dt * self.func(t1, y0, args=None)
        y1 = y0 + dy
        carry = (i+1, t1, dt, y1)
        return (carry, y1)
    
    
    def make_cost_model_feature(self):
        
        def step_fn(carry, inp):
            del inp
            return self.ralston_step_fn(carry)
        
        dummy_t0 = 0.0 
        dummy_dt = 0.1
        dummy_y0 = jnp.ones((self.hidden_size, ))
        carry = (0, dummy_t0, dummy_dt, dummy_y0)
        hlo_module = jax.xla_computation(step_fn)(carry, None).as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        step_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
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
        
        # step flops
        features.append(step_flops)
        # step Arithmetic Intensity
        features.append(step_flops / step_bytes_access)

        total_params = sum(p.size for p in jtu.tree_leaves(eqx.filter(self.ralston_step_fn, eqx.is_array)))

        # total params
        features.append(total_params / 1e6)

        # hidden_size: the dimension of DE
        features.append(self.hidden_size)
        
        # depth
        features.append(self.depth)
        #width_size barrel
        features.append(self.depth)
        # depth of width <= 128
        features.append(0)
        # depth of width <= 256
        features.append(0)
        # depth of width  512
        features.append(0)
        
        return features
        
    def __call__(self, ts, y0, unroll=1):
        t0=ts[0]
        t1=ts[-1]
        dt0=ts[1] - ts[0]
        y0=y0
        carry = (0, t0, dt0, y0)
        
        def step_fn(carry, inp):
            return self.ralston_step_fn(carry)
        
        _, ys = jax.lax.scan(step_fn, carry, xs=None, length=len(ts), unroll=unroll)
        
        return ys
    

def _get_data(ts, *, key):
    y0 = jrandom.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        x = y / (1 + y)
        return jnp.stack([x[1], -x[0]], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


def get_data(dataset_size, num_timesteps, *, key):
    ts = jnp.linspace(0, 10, num_timesteps)
    key = jrandom.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

@eqx.filter_value_and_grad
def grad_loss(model, ti, yi, unroll):
    y_pred = jax.vmap(model, in_axes=(None, 0, None))(ti, yi[:, 0], unroll)
    return jnp.mean((yi - y_pred) ** 2)

@eqx.filter_jit
def make_step(ti, yi, model, optim, opt_state, unroll):
    loss, grads = grad_loss(model, ti, yi, unroll)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def predict_unroll(args):
    key = jrandom.PRNGKey(args.seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)

    ts, ys = get_data(args.dataset_size, args.num_timesteps, key=data_key)
    _, length_size, data_size = ys.shape
    model = NeuralODE(data_size, args.width_size, args.depth, key=model_key)
    
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
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--dataset-size', type=int, default=256)
    parser.add_argument('--width-size', type=int, default=64)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--length', type=int, default=0.5)
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