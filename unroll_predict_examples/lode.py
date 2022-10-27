import time
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

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

matplotlib.rcParams.update({"font.size": 30})

_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6

class Func(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args=None):
        return self.scale * self.mlp(y)


class LatentODE(eqx.Module):
    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int
    
    diffrax_solver: bool
    unroll: int
    depth:int

    def __init__(
        self, *, data_size, hidden_size, latent_size, width_size, depth, key, diffrax_solver=False, unroll=1, **kwargs
    ):
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jrandom.split(key, 5)
        
        self.depth = depth

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)

        self.hidden_to_latent = eqx.nn.Linear(
            hidden_size, 2 * latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)

        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.diffrax_solver = diffrax_solver
        self.unroll = unroll
    # Encoder of the VAE
    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size:]
        std = jnp.exp(logstd)
        latent = mean + jrandom.normal(key, (self.latent_size,)) * std
        return latent, mean, std

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
    
    # Decoder of the VAE
    def _sample(self, ts, latent):
        dt0 = 0.4  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        t0 = ts[0]
        carry = (0, t0, dt0, y0)
        

        def step_fn(carry, input=None):
            del input
            return self.euler_step_fn(carry)
        
        
        if self.diffrax_solver:
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(self.func),
                diffrax.Bosh3(),
                ts[0],
                ts[-1],
                dt0,
                y0,
                saveat=diffrax.SaveAt(ts=ts),
            )
            ys = sol.ys
        else:
            _, ys = jax.lax.scan(step_fn, carry, xs=None,
                                    length=len(ts), unroll=self.unroll)
        return jax.vmap(self.hidden_to_data)(ys)
    
    def make_cost_model_feature(self):
        key = jrandom.PRNGKey(0)
        def step_fn(carry, inp):
            del inp
            return self.ralston_step_fn(carry)
        
        dummy_t0 = 0.0 
        dummy_dt = 0.1
        latent = jrandom.normal(key, (self.latent_size,))
        dummy_y0 = self.latent_to_hidden(latent)
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

        # # total params
        features.append(total_params / 1e6)

        # # hidden_size: the dimension of DE
        features.append(self.hidden_size)
        
        # depth
        features.append(self.depth * 2 + 2)
        #width_size barrel
        features.append(self.depth * 2 + 2)
        # depth of width <= 128
        features.append(0)
        # depth of width <= 256
        features.append(0)
        # depth of width  512
        features.append(0)
        
        return features

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * \
            jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    # Run both encoder and decoder during training.
    def __call__(self, ts, ys, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        return self._loss(ys, pred_ys, mean, std)

    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        latent = jrandom.normal(key, (self.latent_size,))
        return self._sample(ts, latent)


def predict_unroll(args):
    key = jrandom.PRNGKey(args.seed)
    data_key, model_key, loader_key, train_key, sample_key = jrandom.split(key, 5)

    ts, ys = get_data(args.dataset_size, args.num_timesteps, key=data_key)

    model = LatentODE(
        data_size=ys.shape[-1],
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        width_size=args.width_size,
        depth=args.depth,
        key=model_key,
        unroll=args.unroll
    )
    
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
    unroll_list = [1, 2, 5, 10, 20, 25, 40, 50, 100, 200]
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


def get_data(dataset_size, num_timesteps, *, key):
    ykey, tkey1, tkey2 = jrandom.split(key, 3)

    y0 = jrandom.normal(ykey, (dataset_size, 2))

    t0 = 0
    t1 = 2 + jrandom.uniform(tkey1, (dataset_size,))
    ts = jrandom.uniform(tkey2, (dataset_size, num_timesteps)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    def func(t, y, args):
        return jnp.array([[-0.1, 1.3], [-1, -0.1]]) @ y

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    ys = jax.vmap(solve)(ts, y0)

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
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--dataset-size', type=int, default=10000)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--width-size', type=int, default=16)
    parser.add_argument('--latent-size', type=int, default=16)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--num-timesteps', type=int, default=200)
    parser.add_argument('--num-iters', type=int, default=1000)
    parser.add_argument('--unroll', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=5678)
    parser.add_argument('--xgb-dir', type=str, default='../cost-model/ckpt/')

    args = parser.parse_args()
    
    predict_unroll(args)
    
    
if __name__ == "__main__":
    main()