import os
import logging
import argparse
from typing import Sequence, Union, Optional, Any, Dict, Tuple, Callable

import math
import time
from collections import namedtuple
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import optax
import jax
import jax.scipy as jscipy
from jax import numpy as jnp
import jax.random as jrandom
from jax import jit

import flax.linen as nn
from flax.training.train_state import TrainState

Array = jnp.ndarray
Scalar = Union[float, int]

Data = namedtuple('Data', ['ts', 'ts_ext', 'ts_vis', 'ys'])

def prod_diagonal(g, v):
    return g * v

def prod_default(g, v):
    return jnp.einsum('...ij,...j->...i', g, v)

class BaseSDEStep(nn.Module):
    # dim: int
    # layers: Sequence
    dt: float
    noise: str = "default"
    # names: Dict = {
    #     "drift": "f", 
    #     "diffusion": "g", 
    #     "prior_drift": "h"
    # }

    # def method_rename(self, drift="f", diffusion="g", prior_drift="h"):
    #     for name, value in zip(('f', 'g', 'h'),
    #                            (drift, diffusion, prior_drift)):
    #         try:
    #             setattr(self, name, getattr(self, value))
    #         except AttributeError:
    #             pass

    def setup(self):
        if self.noise == "diagonal":
            self.prod_fn = prod_diagonal
        else:
            self.prod_fn = prod_default
        
    def mu_fn(self, t, x):
        raise RuntimeError("Method `drift_fn` has not been provided.")
    
    def sigma_fn(self, t, x):
        raise RuntimeError("Method `diffusion_fn` has not been provided.")

    def __call__(self, carry, dW):
        i, x0 = carry
        t0 = i * self.dt
        drift_term = self.mu_fn(t0, x0) * self.dt
        diffusion_term = self.prod_fn(self.sigma_fn(t0, x0), dW) * jnp.sqrt(self.dt)
        
        x1 = x0 + drift_term + diffusion_term

        carry = (i +1, x1)
        output = x1
        return carry, output


class ForwardSDE(nn.Module):
    sde_step_mdl: nn.Module
    sde_step_kwargs: Dict
    dt: float
    unroll_type: str = "dynamic"
    unroll: int = 1

    @nn.compact
    def __call__(self, y0, dW, ts, times, *args, **kwargs):
        carry = (0, y0)

        if self.unroll_type == "dynamic":
            sdes = nn.scan(self.sde_step_mdl,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=(0),
                out_axes=(0),
                unroll=self.unroll)
            (carry, _ys) = sdes(name="dynamic_sde", **self.sde_step_kwargs)(carry, dW)
        elif self.unroll_type == "static":
            length = dW.shape[0]
            _ys = []
            for i in range(length):
                carry, y = self.sde_step_mdl(name="static_sde", **self.sde_step_kwargs)(carry, dW[i])
                _ys.append(y)
            _ys = jnp.stack(_ys)
        
        # saveat_indices = jnp.searchsorted(times, ts)

        # ys = linear_interpolation(ts[0] + ts_indices * self.dt, 
        #     _ys[ts_indices], 
        #     ts[0] + (ts_indices + 1) * self.dt, 
        #     _ys[ts_indices + 1], 
        #     ts)
        
        dt = self.dt
        
        # ys = linear_interpolation(times[0] + (saveat_indices - 1) * dt, 
        #     _ys[saveat_indices - 1], 
        #     times[0] + (saveat_indices) * dt, 
        #     _ys[saveat_indices], 
        #     saveat)
        return _ys


def _stable_division(a, b, epsilon=1e-6):
    b = jnp.where(jnp.abs(b) > epsilon, b, jnp.full_like(b, fill_value=epsilon) * jnp.sign(b))
    return a / b

def _kl_divergence_normal_normal(p_loc, p_scale, q_loc, q_scale):
    var_ratio = jnp.power(p_scale / q_scale, 2)
    t1 = jnp.power((p_loc - q_loc) / q_scale, 2)
    return 0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio))

class LatentSDEStep(BaseSDEStep):
    theta: float = 1.0
    mu: float = 0.0
    sigma: float = 0.5
    logvar: float = math.log(sigma ** 2 / (2. * theta))

    def setup(self):
        super().setup()
        self.net = nn.Sequential([nn.Dense(256),
            jnp.tanh,
            nn.Dense(256),
            jnp.tanh,
            nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)]
        )

    def f(self, t, y):  # Approximate posterior drift.
        # if not t.shape:
        t = jnp.full_like(y, fill_value=t)
        # Positional encoding in transformers for time-inhomogeneous posterior.
        return self.net(jnp.concatenate((jnp.sin(t), jnp.cos(t), y), axis=-1))

    def g(self, t, y):  # Shared diffusion.
        # print(type(self.sigma))
        # print(y.shape)
        return jnp.full((y.shape[0], 1), self.sigma)

    def h(self, t, y):  # Prior drift.
        return self.theta * (self.mu - y)

    def mu_fn(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = 0.5 * jnp.sum((u ** 2), axis=1, keepdims=True)
        return jnp.concatenate([f, f_logqp], axis=1)

    def sigma_fn(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:1]
        g = self.g(t, y)
        g_logqp = jnp.zeros_like(y)
        return jnp.concatenate([g, g_logqp], axis=1)


class LatentSDE(nn.Module):
    step_mdl: nn.Module
    step_mdl_kwargs: Dict
    theta: float = 1.0
    mu: float = 0.0
    sigma: float = 0.5
    logvar: float = math.log(sigma ** 2 / (2. * theta))

    def setup(self):
        self.py0_mean = jnp.array([[self.mu]])
        self.py0_logvar = jnp.array([[self.logvar]])

        self.qy0_mean = self.param("qy0_mean", nn.initializers.constant(self.mu), (1, 1))
        self.qy0_logvar = self.param("qy0_logvar", nn.initializers.constant(self.logvar), (1, 1))

        self.sde = ForwardSDE(self.step_mdl, self.step_mdl_kwargs, self.step_mdl_kwargs.get("dt", 1e-3))

    @property
    def py0_std(self):
        return jnp.exp(0.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return jnp.exp(0.5 * self.qy0_logvar)

    @nn.compact
    def __call__(self, dW, ts, times, batch_size, rng, unroll=1, *args, **kwargs):
        
        eps = jrandom.normal(rng, (batch_size, 1))
        y0 = self.qy0_mean + eps * self.qy0_std

        logqp0 = jnp.sum(_kl_divergence_normal_normal(self.qy0_mean, self.qy0_std, self.py0_mean, self.py0_std), axis=1)

        aug_y0 = jnp.concatenate([y0, jnp.zeros((batch_size, 1))], axis=1)

        aug_ys = self.sde(aug_y0, dW, ts, times, unroll)
        
        ys, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, 1]
        logqp = jnp.mean(logqp0 + logqp_path, axis=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, dW, ts, times, batch_size, rng=jrandom.PRNGKey(0)):
        eps = jrandom.normal(rng, (batch_size, 1))
        y0 = self.py0_mean + eps * self.py0_std

        return self.sde(y0, dW, ts, times)

    def sample_q(self, dW, ts, times, batch_size, rng=jrandom.PRNGKey(0)):
        eps = jrandom.normal(rng, (batch_size, 1))
        y0 = self.qy0_mean + eps * self.qy0_std
        y0 = jnp.concatenate([y0, jnp.zeros((batch_size, 1))], axis=1)

        return self.sde(y0, dW, ts, times)


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Array):
        x = np.array(x)
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val


def make_segmented_cosine_data():
    ts = jnp.concatenate((jnp.linspace(0.3, 0.8, 10), jnp.linspace(1.3, 1.8, 10)), axis=0)
    ts_ext = jnp.array([0.] + list(ts) + [2.0])
    ts_vis = jnp.linspace(0., 2.0, 300)
    ys = jnp.cos(ts * (2. * math.pi))[:, None]

    return Data(ts, ts_ext, ts_vis, ys)


@partial(jit, static_argnums=[5, 7])
def train_step(step, train_state, bm, ts_ext, times, batch_size, ys, unroll, rng):

    def loss_fn(params):

        @partial(jit, static_argnums=[4])
        def forward_fn(params, bm, ts_ext, times, batch_size, rng):
            zs, kl = train_state.apply_fn({'params': params}, bm, ts_ext, times, batch_size, rng, unroll=unroll)
            return zs, kl
        
        zs, kl = forward_fn(params, bm, ts_ext, times, batch_size, rng)
        # zs, kl = train_state.apply_fn({'params': params}, bm, ts_ext, times, batch_size, rng)

        zs = jnp.squeeze(zs)
        zs = zs[1:-1]

        logpy = jnp.mean(jnp.sum(jscipy.stats.laplace.logpdf(x=ys, loc=zs, scale=0.05), axis=0), axis=0)
        kl_scheduler = jnp.min(jnp.array([1.0, 1.0 / 100 * step]))
        
        loss = -logpy + kl * kl_scheduler

        return (loss, (logpy, kl))
    
    (loss, (logpy, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss, logpy, kl

def main(args):

    vis_batch_size = 1024
    ylims = (-1.75, 1.75)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    vis_idx = np.random.permutation(vis_batch_size)
    # From https://colorbrewer2.org/.
    if args.color == "blue":
        sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
        fill_color = '#9ebcda'
        mean_color = '#4d004b'
        num_samples = len(sample_colors)
    else:
        sample_colors = ('#fc4e2a', '#e31a1c', '#bd0026')
        fill_color = '#fd8d3c'
        mean_color = '#800026'
        num_samples = len(sample_colors)

    batch_size = 512
    rng = jrandom.PRNGKey(42)

    ts, ts_ext, ts_vis, ys = make_segmented_cosine_data()

    dt = 1e-2
    num_timesteps = math.floor((ts_ext[-1] - ts_ext[0]) / dt)
    bm = jrandom.normal(rng, (num_timesteps, batch_size, 2))
    times = jnp.arange(ts_ext[0], ts_ext[-1], dt)

    # make features
    output = ""
    cell = LatentSDEStep(dt=dt, noise='diagonal')
    dummy_carry = (0, jnp.zeros((batch_size, 2)))
    cell_variable = cell.init(rng, carry=dummy_carry, dW=bm[0])

    hlo_module = jax.xla_computation(cell.apply)({'params': cell_variable['params']}, carry=dummy_carry, dW=bm[0]).as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
    bytes_access_gb = cost['bytes accessed'] / 1e9
    flops_g = cost['flops'] / 1e9
    output = output + str(bytes_access_gb) + ','
    output = output + str(flops_g) + ','
    output = output + str(flops_g / bytes_access_gb) + ','
    output = output + str(2) + ','
    # params share across different iteration, 0 false
    output = output + str(0) + ','
    
    total_params = sum(p.size for p in jax.tree_leaves(cell_variable['params']))
    # million params
    output = output + str(total_params / 1e6) + ','
    output = output + str(args.batch_size) + ","
    output = output + str(args.unroll) + ","
    output = output + str(num_timesteps) + ","


    mdl_kwargs = {'dt': dt, 'noise': "diagonal"}
    latent_sde = LatentSDE(LatentSDEStep, mdl_kwargs, dt)

    variables = latent_sde.init(rng, bm, ts, times, batch_size, rng)

    learning_rate = 1e-2
    learning_rate_fn = optax.exponential_decay(learning_rate, 1, 0.999)
    tx = optax.adam(learning_rate=learning_rate_fn)

    state = TrainState.create(
        apply_fn=latent_sde.apply,
        params=variables['params'],
        tx=tx
    )

    if args.show_prior:
        zs = latent_sde.apply({'params': state.params}, dW=bm, ts=ts_vis, times=times, batch_size=batch_size, rng=rng, method=latent_sde.sample_p)
        zs = jnp.sort(zs, axis=1)

        img_dir = os.path.join(args.train_dir, 'prior.png')
        plt.subplot(frameon=False)
        for alpha, percentile in zip(alphas, percentiles):
            idx = int((1 - percentile) / 2. * vis_batch_size)
            zs_bot_ = zs[:, idx]
            zs_top_ = zs[:, -idx]
            plt.fill_between(ts_vis, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

        # `zorder` determines who's on top; the larger the more at the top.
        plt.scatter(ts, ys, marker='x', zorder=3, color='k', s=35)  # Data.
        plt.ylim(ylims)
        plt.xlabel('$t$')
        plt.ylabel('$Y_t$')
        plt.tight_layout()
        plt.savefig(img_dir, dpi=args.dpi)
        plt.close()
        logging.info(f'Saved prior figure at: {img_dir}')

    start_time = time.time()
    iter_time_list = []
    logpy_metric = EMAMetric()
    kl_metric = EMAMetric()
    loss_metric = EMAMetric()
    unroll = args.unroll
    # print(unroll)

    for step in range(args.train_iters):
        rng, _ = jax.random.split(rng)
        bm = jrandom.normal(rng, (num_timesteps, batch_size, 2))
        state, loss, logpy, kl = train_step(step, state, bm, ts_ext, times, batch_size, ys, unroll, rng)
        
        logpy_metric.step(logpy)
        kl_metric.step(kl)
        loss_metric.step(loss)

        if step == 0:
            compile_time = time.time()
            iter_time = time.time()
        # if step % 100 == 0 and step > 0:
        #     iter_time_list.append(time.time() - iter_time)
        #     iter_time = time.time()

        if step % args.pause_iters == 0:
            
            if args.plot:
                img_path = os.path.join(args.train_dir, f'global_step_{step}.png')

                zs = latent_sde.apply({'params': state.params}, dW=bm, ts=ts_vis, times=times, batch_size=batch_size, rng=rng, method=latent_sde.sample_q).squeeze()
                zs = zs[:, :, 0:1]
                samples = zs[:, vis_idx]
                ts_vis_, zs_, samples_ = np.array(ts_vis), np.array(zs), np.array(samples)
                zs_ = np.sort(zs_, axis=1)
                plt.subplot(frameon=False)

                if args.show_percentiles:
                    for alpha, percentile in zip(alphas, percentiles):
                        idx = int((1 - percentile) / 2. * vis_batch_size)
                        zs_bot_, zs_top_ = zs_[:, idx], zs_[:, -idx]
                        
                        zs_bot_ = np.squeeze(zs_bot_)
                        zs_top_ = np.squeeze(zs_top_)
                        
                        plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

                if args.show_mean:
                    plt.plot(ts_vis_, zs_.mean(axis=1), color=mean_color)

                if args.show_samples:
                    for j in range(num_samples):
                        plt.plot(ts_vis_, samples_[:, j], color=sample_colors[j], linewidth=1.0)

                if args.hide_ticks:
                    plt.xticks([], [])
                    plt.yticks([], [])

                plt.scatter(np.array(ts), np.array(ys), marker='x', zorder=3, color='k', s=35)  # Data.
                plt.ylim(ylims)
                plt.xlabel('$t$')
                plt.ylabel('$Y_t$')
                plt.tight_layout()
                plt.savefig(img_path, dpi=args.dpi)
                plt.close()

            iter_time = time.time()
            # print(
            #     f'global_step: {step}, '
            #     f'logpy: {logpy_metric.val:.3f}, '
            #     f'kl: {kl_metric.val:.3f}, '
            #     f'loss: {loss_metric.val:.3f}'
            # )
    output = output + str(compile_time - start_time) + ','
    output = output + str(time.time() - compile_time)
    print(output)

def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--debug', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-dir', type=str, default='./logs', required=False)
    parser.add_argument('--save-ckpt', type=str2bool, default=False, const=True, nargs="?")

    parser.add_argument('--data', type=str, default='segmented_cosine', choices=['segmented_cosine', 'irregular_sine'])
    parser.add_argument('--kl-anneal-iters', type=int, default=100, help='Number of iterations for linear KL schedule.')
    parser.add_argument('--train-iters', type=int, default=1000, help='Number of iterations for training.')
    parser.add_argument('--pause-iters', type=int, default=50, help='Number of iterations before pausing.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--likelihood', type=str, choices=['normal', 'laplace'], default='laplace')
    parser.add_argument('--scale', type=float, default=0.05, help='Scale parameter of Normal and Laplace.')

    parser.add_argument('--adjoint', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--adaptive', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--method', type=str, default='euler', choices=('euler', 'milstein', 'srk'),
                        help='Name of numerical solver.')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--unroll', type=int, default=1)

    parser.add_argument('--plot', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-prior', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-samples', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-percentiles', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-arrows', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-mean', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--hide-ticks', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--color', type=str, default='blue', choices=('blue', 'red'))
    args = parser.parse_args()

    main(args)