import time
import math
from typing import Union, Sequence
from dataclasses import dataclass
from functools import partial

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

from jax.config import config
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
    sf: jnp.ndarray  # diffusion
    noise_size: int

    def __init__(
        self,
        noise_size,
        hidden_size,
        width_size,
        depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mf_key, sf_key = jrandom.split(key, 2)

        self.mf = MuField(hidden_size, width_size, depth, key=mf_key)
        self.sf = jnp.full((hidden_size, noise_size), 0.3)

        self.noise_size = noise_size

    def __call__(self, carry, input=None):
        (i, t0, dt, y0, key) = carry
        t = jnp.full((1, ), t0 + i * dt)
        _key1, _key2 = jrandom.split(key, 2)
        bm = jrandom.normal(_key1, (self.noise_size, )) * jnp.sqrt(dt)
        drift_term = self.mf(t=t, y=y0) * dt
        diffusion_term = jnp.dot(self.sf, bm)
        y1 = y0 + drift_term + diffusion_term
        carry = (i+1, t0, dt, y1, _key2)

        return carry, y1

class NeuralSDE(eqx.Module):
    step: SDEStep
    noise_size: int
    hidden_size: int
    depth: int
    width_size: int


    def __init__(
        self,
        noise_size,
        hidden_size,
        width_size,
        depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        step_key, _ = jrandom.split(key, 2)

        self.step = SDEStep(noise_size=noise_size, hidden_size=hidden_size, width_size=width_size, depth=depth, key=step_key)

        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.width_size = width_size
        self.depth = depth

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
        # step bytes access
        features.append(step_bytes_access)
        features.append(step_bytes_access_op0)
        features.append(step_bytes_access_op1)
        features.append(step_bytes_access_out)

        # step FLOPS 
        features.append(step_flops)
        # step Arithmetic Intensity
        features.append(step_flops / step_bytes_access)

        total_params = sum(p.size for p in jax.tree_leaves(eqx.filter(self.step, eqx.is_array)))

        # total params
        features.append(total_params / 1e6)

        # hidden_size: the dimension of DE
        features.append(self.hidden_size)

        # noise_size: browian motion size ? 
        # TODO should we add this for ODE/CDE？
        # output = output + str(self.noise_size) + ','
        
        # width_size: width for every layer of MLP
        # output = output + str(self.width_size) + ','
        
        # depth: depth of MLP
        features.append(self.depth * 2)

        return features

    def __call__(self, y0, t0, dt, num_timesteps, unroll, key):

        _, bm_key = jrandom.split(key, 2)

        def step_fn(carry, inp):
            return self.step(carry, inp)
        
        carry = (0, t0, dt, y0, bm_key)

        # _, ys = jax.lax.scan(step_fn, carry, xs=None, length=num_timesteps, unroll=unroll)
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
            args.width_size,
            args.depth,
            key=key,
        )

    features = model.make_cost_model_feature()
    features.append(args.batch_size)
    features.append(args.num_timesteps)
    features.append(args.unroll)

    y0 = jnp.ones((args.batch_size, args.hidden_size))

    learning_rate = 1e-2
    learning_rate_fn = optax.exponential_decay(learning_rate, 1, 0.999)
    optimizer = optax.adam(learning_rate=learning_rate_fn)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    start_time = time.time()

    for step in range(1000):
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
    features = features + str(args.batch_size) + ','
    features = features + str(args.unroll) + ','
    output = features + str(compile_time - start_time) + ','
    output = output + str(time.time() - compile_time)
    print(output)

@dataclass
class Args:
    batch_size: int

    # dim of SDE
    hidden_size: int
    noise_size: int 
    num_timesteps: int
    num_iters: int
    
    # network
    depth: Sequence[int]
    width_size: int
    
    # dynamic unroll
    unroll: int
    T: float = 1.0


def main(args):
    train(args)


if __name__ == '__main__':
    # test code
    args = Args(batch_size=128, 
                        hidden_size=16,
                        noise_size=16,
                        num_timesteps=50,
                        num_iters=1000, 
                        depth=3, 
                        width_size=64,
                        unroll=1)
    # warm up run
    main(args=args)
    # for batch_size in [512]:
    for batch_size in [128, 256, 512]:
        # for num_timesteps in [50]:
        for num_timesteps in [50, 100, 200]:
            # for width_size in [64]:
            for width_size in [64, 128, 256, 512, 1024]:
                # for depth in [3]:
                for depth in [3, 4, 5, 6]:
                    for hidden_size in [16, 32, 64]:
                        n = 0
                        while n <= 5:
                            if n == 0:
                                unroll = 1
                            else:
                                unroll = math.ceil(0.1 * n * num_timesteps)
                                if unroll > 100:
                                    break
                            args = Args(batch_size=batch_size, 
                                hidden_size=hidden_size,
                                noise_size=hidden_size,
                                num_timesteps=num_timesteps,
                                num_iters=1000, 
                                depth=depth, 
                                width_size=width_size,
                                unroll=unroll)
                            n += 1
                            main(args=args)