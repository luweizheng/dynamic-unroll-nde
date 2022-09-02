from dataclasses import dataclass
import functools
import math
import time
from diffrax.misc import ω
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import jax.tree_util as jtu
import optax  # https://github.com/deepmind/optax
import xgboost as xgb
import sys; 
sys.path.insert(0, '..')
from simulated_annealing import annealing
import numpy as np

@dataclass
class Args:
    
    batch_size: int
    lr:float
    # dim of SDE
    dataset_size: int
    add_noise: bool
    
    num_timesteps: int
    num_iters: int
    
    # network
    hidden_size:int
    depth: int
    width_size: int
    
    # dynamic unroll
    unroll: int 
    seed:int
    search_method: str = "exhaustive"
    
    # dataset_size=256,
    # add_noise=False,
    # batch_size=32,
    # lr=1e-2,
    # steps=20,
    # hidden_size=8,
    # width_size=128,
    # depth=1,
    # seed=5678,

class Func(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int
    depth: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)

    
class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear
    hidden_size: int
    width_size: int
    depth: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jrandom.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)
        self.hidden_size = hidden_size
        self.width_size = width_size
        self.depth = depth
    
    def make_cost_model_feature(self, num_timesteps):
        key = jrandom.PRNGKey(5678)
        ts, coeffs, _ ,_ = get_data(256, False, num_timesteps, key=key)
        ts = ts[0]
        coeffs = jnp.asarray(coeffs)
        coeffs = coeffs[:,0,:]
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        def step_fn(carry, inp):
            del inp
            return self.step(carry, term)
        
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

        total_params = sum(p.size for p in jtu.tree_leaves(eqx.filter(self.step, eqx.is_array)))

        # total params
        features.append(total_params / 1e6)

        # hidden_size: the dimension of DE
        features.append(self.hidden_size)
        
        # depth
        features.append(self.depth)
        #width_size barrel
        # depth of width <=128
        features.append(self.depth)
        # depth of width <= 256
        features.append(0)
        # depth of width <= 512
        features.append(0)
        # depth of width > 512
        features.append(0)
        
        return features
        
    def step(self, carry, term):
        (i, t0, dt, y0) = carry
        control = dt
        y1 = (y0**ω + term.vf_prod(t0, y0, args=None, control=control) ** ω).ω
        t1 = t0 + dt
        carry = (i+1, t1, dt, y1)
        return (carry , y1)

    def __call__(self, ts, coeffs, evolving_out=False, unroll=1):
        
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        dt0 = ts[1] - ts[0]
        y0 = self.initial(control.evaluate(ts[0]))
        carry = (0, ts[0], dt0, y0)
        
        def step_fn(carry, inp=None):
            del inp
            return self.step(carry, term)
        
        _, ys = jax.lax.scan(step_fn, carry, xs=None, length=len(ts), unroll=unroll)
        if evolving_out:
            prediction = jax.vmap(lambda y: jnn.sigmoid(self.linear(y))[0])(ys)
        else:
            (prediction,) = jnn.sigmoid(self.linear(ys[-1]))
        return prediction

def get_data(dataset_size, add_noise, num_timesteps, *, key):
    theta_key, noise_key = jrandom.split(key, 2)
    length = num_timesteps
    theta = jrandom.uniform(theta_key, (dataset_size,), minval=0, maxval=2 * math.pi)
    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    ts = jnp.broadcast_to(jnp.linspace(0, 4 * math.pi, length), (dataset_size, length))
    matrix = jnp.array([[-0.3, 2], [-2, -0.3]])
    ys = jax.vmap(
        lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
    )(y0, ts)
    ys = jnp.concatenate([ts[:, :, None], ys], axis=-1)  # time is a channel
    ys = ys.at[: dataset_size // 2, :, 1].multiply(-1)
    if add_noise:
        ys = ys + jrandom.normal(noise_key, ys.shape) * 0.1
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    labels = jnp.zeros((dataset_size,))
    labels = labels.at[: dataset_size // 2].set(1.0)
    _, _, data_size = ys.shape
    return ts, coeffs, labels, data_size

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
    


def train(args):
    key = jrandom.PRNGKey(args.seed)
    train_data_key, test_data_key, model_key, loader_key = jrandom.split(key, 4)

    ts, coeffs, labels, data_size = get_data(
        args.dataset_size, args.add_noise, args.num_timesteps,key=train_data_key
    )
    
    model = NeuralCDE(data_size, args.hidden_size, args.width_size, args.depth, key=model_key)
    
    features = model.make_cost_model_feature(args.num_timesteps)
    features.append(args.batch_size)
    features.append(args.num_timesteps // 5)
    
    compile_model_loaded = xgb.Booster()
    compile_model_loaded.load_model("../cost-model/ckpt/titan_compile.txt")
    run_model_loaded = xgb.Booster()
    run_model_loaded.load_model("../cost-model/ckpt/titan_execution.txt")
    
    
    
    def cost_fn(unroll):
        cur_features = features + [unroll]
        
        compilation_time_pred = compile_model_loaded.predict(xgb.DMatrix([cur_features]))
        run_time_pred = run_model_loaded.predict(xgb.DMatrix([cur_features]))
        total_time_pred = compilation_time_pred + run_time_pred * 5
        
        return total_time_pred
    
    if args.search_method == "exhaustive":
        # exhaustively iterate a list of candidates
        unroll_list = [2, 5, 8, 10, 20, 40, 50]
        total_time_pred_list = []
        for unroll in unroll_list:
            total_time_pred = cost_fn(unroll)
            total_time_pred_list.append(total_time_pred)
        predicted_unroll = unroll_list[np.argmin(total_time_pred_list)]
    elif args.search_method == "sa_scipy":
        # dual annealing from scipy
        bounds = [[2, args.num_timesteps // 5]]
        from scipy.optimize import dual_annealing

        result = dual_annealing(cost_fn, bounds, maxiter=20)
        predicted_unroll = result['x']
    elif args.search_method == "sa_our":
        # my own implementation of SA
        bounds = (2, args.num_timesteps // 5)

        def clip(x, bounds):
            """ Force x to be in the interval."""
            a, b = bounds
            return int(max(min(x, b), a))

        def random_neighbour(x, bounds, fraction=1):
            """Move a little bit x, from the left or the right."""
            amplitude = (max(bounds) - min(bounds)) * fraction / 10
            delta = (-amplitude/2.) + amplitude * np.random.random_sample()
            return clip(x + delta, bounds)
        predicted_unroll, _, _, _ = annealing(bounds, cost_fn, random_neighbour=random_neighbour, maxsteps=20, debug=False)
    
    print(f"predicted unroll: {predicted_unroll}")
    
    
    @eqx.filter_jit
    def loss(model, ti, label_i, coeff_i, unroll):
        fn = functools.partial(model, unroll=unroll)
        pred = jax.vmap(fn)(ti, coeff_i)
        # Binary cross-entropy
        bxe = label_i * jnp.log(pred) + (1 - label_i) * jnp.log(1 - pred)
        bxe = -jnp.mean(bxe)
        acc = jnp.mean((pred > 0.5) == (label_i == 1))
        return bxe, acc

    grad_loss = eqx.filter_value_and_grad(loss, has_aux=True)

    @eqx.filter_jit
    def make_step(model, data_i, opt_state):
        ti, label_i, *coeff_i = data_i
        (bxe, acc), grads = grad_loss(model, ti, label_i, coeff_i, args.unroll)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return bxe, acc, model, opt_state

    optim = optax.adam(args.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    start_time = time.time()
    for step, data_i in zip(
        range(args.num_iters), dataloader((ts, labels) + coeffs, args.batch_size, key=loader_key)
    ):
        bxe, acc, model, opt_state = make_step(model, data_i, opt_state)
    total_time = time.time() - start_time
    print(f"unroll: {args.unroll}, actuall time: {total_time}")

    del model

def main():
    unroll_list = [1, 2, 5, 8, 10, 20, 40, 50]
    args = Args(batch_size=64,
                lr=1e-2,
                dataset_size=256,
                add_noise=False,
                num_timesteps=1000,
                num_iters=500,
                hidden_size=32,
                depth=3,
                width_size=128,
                unroll=1,
                seed=5678)
    
    #warm up
    train(args)
    args.search_method = "sa_scipy"
    for unroll in unroll_list:
        args.unroll = unroll
        train(args)
        args.search_method = "sa_our"
    
if __name__ == '__main__':
    main()