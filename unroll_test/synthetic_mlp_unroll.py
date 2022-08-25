import time
import math
import argparse
from typing import Union, Sequence
from dataclasses import dataclass
from functools import partial

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax  # https://github.com/deepmind/optax

from jax.config import config
import itertools
import jax.tree_util as jtu
# We use GPU as the default backend.
# If you want to use cpu as backend, uncomment the following line.
# config.update("jax_platform_name", "cpu")

def lipswish(x):
    return 0.909 * jnn.silu(x)


class MuField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        _, mlp_key = jrandom.split(key)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.relu,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t, y):
        return self.mlp(jnp.concatenate([t, y]))


class SigmaField(eqx.Module):
    mlp: eqx.nn.MLP
    noise_size: int
    hidden_size: int

    def __init__(
        self, noise_size, hidden_size, width_size, depth, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        _, mlp_key = jrandom.split(key)

        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * noise_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.noise_size = noise_size
        self.hidden_size = hidden_size

    def __call__(self, t, y):
        return self.mlp(jnp.concatenate([t, y])).reshape(
            self.hidden_size, self.noise_size
        )


class SDEStep(eqx.Module):
    mf: MuField  # drift
    sf: SigmaField  # diffusion
    noise_size: int

    def __init__(
        self,
        noise_size,
        hidden_size,
        mu_width_size,
        sigma_width_size,
        mu_depth,
        sigma_depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mf_key, sf_key = jrandom.split(key, 2)

        self.mf = MuField(hidden_size, mu_width_size, mu_depth, key=mf_key)
        self.sf = SigmaField(
            noise_size, hidden_size, sigma_width_size, sigma_depth, key=sf_key
        )

        self.noise_size = noise_size

    def __call__(self, carry, input=None):
        (i, t0, dt, y0, key) = carry
        t = jnp.full((1, ), t0 + i * dt)
        _key1, _key2 = jrandom.split(key, 2)
        bm = jrandom.normal(_key1, (self.noise_size, )) * jnp.sqrt(dt)
        drift_term = self.mf(t=t, y=y0) * dt
        diffusion_term = jnp.dot(self.sf(t=t, y=y0), bm)
        y1 = y0 + drift_term + diffusion_term
        carry = (i+1, t0, dt, y1, _key2)

        return carry, y1

class NeuralSDE(eqx.Module):
    step: SDEStep
    noise_size: int
    hidden_size: int
    mu_depth: int
    sigma_depth: int
    mu_width_size: int
    sigma_width_size: int


    def __init__(
        self,
        noise_size,
        hidden_size,
        mu_width_size,
        sigma_width_size,
        mu_depth,
        sigma_depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        step_key, _ = jrandom.split(key, 2)

        self.step = SDEStep(noise_size=noise_size,
            hidden_size=hidden_size,
            mu_width_size=mu_width_size,
            sigma_width_size=sigma_width_size,
            mu_depth=mu_depth,
            sigma_depth=sigma_depth, 
            key=step_key)

        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.mu_width_size = mu_width_size
        self.sigma_width_size = sigma_width_size
        self.mu_depth = mu_depth
        self.sigma_depth = sigma_depth


    def make_cost_model_feature(self):

        def step_fn(carry, inp):
            return self.step(carry, inp)

        dummy_t0 = 0.0
        dummy_dt = 0.1

        dummy_y0 = jnp.ones((self.hidden_size, ))
        dummy_bm_key = jrandom.PRNGKey(0)
        carry = (0, dummy_t0, dummy_dt, dummy_y0, dummy_bm_key)
        hlo_module = jax.xla_computation(step_fn)(carry, None).as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        
        step_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
        step_bytes_access = step_cost['bytes accessed']
        step_bytes_access_op0 = step_cost['bytes accessed operand 0 {}']
        step_bytes_access_op1 = step_cost['bytes accessed operand 1 {}']
        step_bytes_access_out = step_cost['bytes accessed output {}']
        step_flops = step_cost['flops']
        
        features = []
        # f0: step bytes access
        features.append(step_bytes_access)
        # f1: op0 bytes access
        features.append(step_bytes_access_op0)
        # f2: op1 bytes access
        features.append(step_bytes_access_op1)
        # f3: out bytes access
        features.append(step_bytes_access_out)

        # f4: step FLOPS 
        features.append(step_flops)
        # f5: step Arithmetic Intensity
        features.append(step_flops / step_bytes_access)

        total_params = sum(p.size for p in jtu.tree_leaves(eqx.filter(self.step, eqx.is_array)))

        # f6: total params
        features.append(total_params / 1e6)

        # f7: the dimension of DE
        features.append(self.hidden_size)
        
        # f8: depth of all MLP, in this case, mu and sigma
        features.append(self.mu_depth + self.sigma_depth)

        # count of width
        w128=0
        w256=0
        w512=0
        w512lg=0
        for d, w in zip([self.mu_depth, self.sigma_depth], [self.mu_width_size, self.sigma_width_size]):
            if w <= 128:
                w128 += d
            elif w <= 256:
                w256 += d
            elif w <= 512:
                w512 += d
            else:
                w512lg += d
        
        # f9: width <= 128
        features.append(w128)
        # f10: 128 < width <= 256
        features.append(w256)
        # f11: 256 < width <= 512
        features.append(w512)
        # f12: width > 512
        features.append(w512lg)

        return features

    def __call__(self, y0, t0, dt, num_timesteps, unroll, key):

        _, bm_key = jrandom.split(key, 2)

        def step_fn(carry, inp):
            return self.step(carry, inp)
        
        ys = solve(step_fn, y0, t0, dt, num_timesteps, unroll, bm_key)
        
        return ys



def solve(step, y0, t0, dt, num_timesteps, unroll, bm_key):
    carry = (0, t0, dt, y0, bm_key)

    _, ys = jax.lax.scan(step, carry, xs=None, length=num_timesteps, unroll=unroll)

    return ys

@eqx.filter_jit
def loss_fn(model, y0, t0, dt, num_timesteps, unroll, key):

    ys = jax.vmap(model, in_axes=[0, None, None, None, None, None])(y0, t0, dt, num_timesteps, unroll, key)
    
    # dummy loss
    loss = jnp.sum(jnp.mean(ys, axis=0))

    return loss


@eqx.filter_value_and_grad
def grad_loss(model, y0, t0, dt, num_timesteps, unroll, key):
    return loss_fn(model, y0, t0, dt, num_timesteps, unroll, key)


@eqx.filter_jit
def train_step(model, y0, t0, dt, num_timesteps, optimizer, opt_state, unroll, key):
   
    loss, grads = grad_loss(model, y0, t0, dt, num_timesteps, unroll, key)

    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, model

def train(args):

    key = jrandom.PRNGKey(42)

    model = NeuralSDE(
            args.noise_size,
            args.hidden_size,
            args.mu_width_size,
            args.sigma_width_size,
            args.mu_depth,
            args.sigma_depth,
            key=key,
        )
    
    features = []

    # features = model.make_cost_model_feature()
    # # f13: batch size
    # features.append(args.batch_size)
    # # f14: num of time steps
    # features.append(args.num_timesteps)
    # f15: unroll
    features.append(args.unroll)

    y0 = jnp.ones((args.batch_size, args.hidden_size))
    learning_rate = 1e-2
    learning_rate_fn = optax.exponential_decay(learning_rate, 1, 0.999)
    optimizer = optax.adam(learning_rate=learning_rate_fn)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    start_time = time.time()

    for step in range(args.num_iters):
        key, _ = jax.random.split(key)
        loss = train_step(model, y0, 0, 0.1, args.num_timesteps, optimizer, opt_state, unroll=args.unroll, key=key)

        if step == 0:
            compile_time = time.time()
            # iter_time = time.time()
        # print(f"iter: {time.time() - iter_time}")
        # iter_time = time.time()
        # if step % 100 == 0 and step > 0:
        #     iter_time_list.append(time.time() - iter_time)
        #     iter_time = time.time()
    
    features.append(compile_time - start_time)
    features.append(time.time() - compile_time)
    features.append(time.time() - start_time)
    print(','.join(map(str, features)))

@dataclass
class Args:
    batch_size: int

    # dim of SDE
    hidden_size: int
    noise_size: int 
    num_timesteps: int
    num_iters: int
    
    # network
    mu_depth: int
    mu_width_size: int
    sigma_depth: int
    sigma_width_size: int
    
    # dynamic unroll
    unroll: int
    T: float = 1.0


def main():
    print("unroll, compile_time, execute_time, total_time")
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    
    cli_args = parser.parse_args()
    batch_size = cli_args.batch_size
    num_timesteps = cli_args.num_timesteps
    # warm up run
    args = Args(batch_size=batch_size, 
            hidden_size=64,
            noise_size=64,
            num_timesteps=num_timesteps,
            num_iters=1000, 
            mu_depth=3,
            mu_width_size=64,
            sigma_depth=3,
            sigma_width_size=64,
            unroll=1)
    # dummy run
    train(args)

    unroll_list = [1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 80, 100]
    
    for unroll in unroll_list:
        args.unroll = unroll
        train(args)


if __name__ == '__main__':
    main()