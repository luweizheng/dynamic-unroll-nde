from dataclasses import dataclass
import argparse
from turtle import width
from typing import Sequence, Union, Optional, Any, Dict, Tuple, Callable

import math
import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import optax
import jax
import jax.scipy as jscipy
from jax import numpy as jnp
import jax.random as jrandom
from jax import jit
import jax.nn as jnn

import flax.linen as nn
from flax.training.train_state import TrainState

from jax.config import config
# We use GPU as the default backend.
# If you want to use cpu as backend, uncomment the following line.
# config.update("jax_platform_name", "cpu")

Array = jnp.ndarray
Scalar = Union[float, int]

def prod_diagonal(g, v):
    return g * v

def prod_default(g, v):
    return jnp.einsum('...ij,...j->...i', g, v)

class MLP(nn.Module):
    in_size: int
    out_size: int
    width_size: int
    depth: int
    activation: Callable = nn.relu
    final_activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        layers = []
        if self.depth == 0:
            layers.append(nn.Dense(features=self.out_size))
            layers.append(self.final_activation)
        else:
            layers.append(nn.Dense(features=self.width_size))
            for i in range(self.depth - 1):
                layers.append(nn.Dense(features=self.width_size))
                layers.append(self.activation)
            layers.append(nn.Dense(features=self.out_size))
            layers.append(self.final_activation)
        layers = nn.Sequential(layers)

        x = layers(x)
        return x
        

class MuField(nn.Module):
    hidden_size: int
    width_size: int
    depth: int

    def setup(self):
        self.mlp = MLP(
            in_size=self.hidden_size + 1,
            out_size=self.hidden_size,
            width_size=self.width_size,
            depth=self.depth,
            activation=jnn.relu,
            final_activation=jnn.tanh
        )


    @nn.compact
    def __call__(self, t, y):
        x = jnp.concatenate([t, y], axis=-1)
        x = self.mlp(x)
        return x


class SDEStep(nn.Module):
    noise_size: int
    hidden_size: int
    width_size: int
    depth: int

    def setup(self):
        self.mf = MuField(self.hidden_size, self.width_size, self.depth)
        self.sf = jnp.full((self.noise_size, self.noise_size), 0.3)

    # def __call__(self, carry, input=None):
    #     (i, t0, dt, y0, key) = carry
    #     t = jnp.full((1, ), t0 + i * dt)
    #     _key1, _key2 = jrandom.split(key, 2)
    #     bm = jrandom.normal(_key1, (self.noise_size, )) * jnp.sqrt(dt)
    #     drift_term = self.mf(t=t, y=y0) * dt
    #     diffusion_term = jnp.dot(self.sf, bm)
    #     y1 = y0 + drift_term + diffusion_term
    #     carry = (i+1, t0, dt, y1, _key2)

    #     return carry, y1
    
    def __call__(self, carry, inp):
        i, dt, x0 = carry
        t0 = jnp.full((1, ), i * dt)

        drift_term = self.mf(t0, x0) * dt
        dW = jrandom.normal(jrandom.PRNGKey(0), (self.noise_size, ))

        diffusion_term = jnp.dot(self.sf, dW) * jnp.sqrt(dt)
        x1 = x0 + drift_term + diffusion_term

        carry = (i + 1, dt, x1)
        output = x1
        return carry, output
    

class NeuralDE(nn.Module):
    noise_size: int
    hidden_size: int
    width_size: int
    depth: int
    unroll: int = 1
    dt: float = 0.1

    # def setup(self):
        # self.step = SDEStep(noise_size=self.noise_size, hidden_size=self.hidden_size, width_size=self.width_size, depth=self.depth)

    @nn.compact
    def __call__(self, y0, dW, num_timesteps, *args, **kwargs):
        carry = (0, self.dt, y0)
        sdes = nn.scan(SDEStep,
            length=num_timesteps,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            unroll=self.unroll)
        (carry, _ys) = sdes(name="dynamic_sde", noise_size=self.noise_size, hidden_size=self.hidden_size, width_size=self.width_size, depth=self.depth)(carry, dW)
        
        return _ys


@partial(jit)
def train_step(step, train_state, y0, dW):

    def loss_fn(params):

        @partial(jax.vmap, in_axes=[None, 0])
        def forward_fn(params, y0):
            ys = train_state.apply_fn({'params': params}, y0, None, 64)
            return ys

        ys = forward_fn(params, y0)
        
        # dummy loss
        loss = jnp.sum(jnp.mean(ys, axis=0))

        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss


def train():

    rng = jrandom.PRNGKey(42)

    noise_size = 16
    hidden_size = 16
    width_size = 256
    depth = 6
    unroll = 1
    key = jrandom.PRNGKey(0)

    model = NeuralDE(
            noise_size,
            hidden_size,
            width_size,
            depth,
            unroll
        )

    dummy_y0 = jnp.ones((hidden_size, ))
    num_timesteps = 64

    variables = model.init(key, y0=dummy_y0, dW=None, num_timesteps=num_timesteps)

    # make features
    output = ""
    # cell = SDEStep(dim=args.dim, layers=args.layers, dt=dt, noise='diagonal')

    # dummy_carry = (0, jnp.zeros((args.batch_size, args.dim)))
    # cell_variable = cell.init(rng, carry=dummy_carry, dW=dummy_dW[0])

    # hlo_module = jax.xla_computation(cell.apply)({'params': cell_variable['params']}, carry=dummy_carry, dW=dummy_dW[0]).as_hlo_module()
    # client = jax.lib.xla_bridge.get_backend()
    # step_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
    # step_bytes_access_gb = step_cost['bytes accessed'] / 1e9
    # step_flops_g = step_cost['flops'] / 1e9
    
    # # step bytes access in GB
    # output = output + str(step_bytes_access_gb) + ','
    # # step G FLOPS 
    # output = output + str(step_flops_g) + ','
    # # step Arithmetic Intensity
    # output = output + str(step_flops_g / step_bytes_access_gb) + ','

    
    # total_params = sum(p.size for p in jax.tree_leaves(cell_variable['params']))

    # mdl_kwargs = {'dim': args.dim, 'dt': dt, 'noise': "diagonal", 'layers': args.layers}
    # sde = ForwardSDE(SDEStep, mdl_kwargs, dt, unroll=args.unroll)

    # variables = sde.init(rng, y0, dummy_dW, ts, times)

    # hlo_module = jax.xla_computation(sde.apply)({'params': variables['params']}, y0, dummy_dW, ts, times).as_hlo_module()
    # full_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
    # full_bytes_access_gb = full_cost['bytes accessed'] / 1e9
    # full_flops_g = full_cost['flops'] / 1e9
    # # step bytes access in GB
    # output = output + str(full_bytes_access_gb) + ','
    # # step G FLOPS 
    # output = output + str(full_flops_g) + ','
    # # step Arithmetic Intensity
    # output = output + str(full_flops_g / full_bytes_access_gb) + ','

    # #  params size in million
    # output = output + str(total_params / 1e6) + ','

    # # input dimension
    # output = output + str(args.dim) + ','
    # # number of layers
    # output = output + str(len(args.layers)) + ','

    # # device 1 Tesla V100 2 Titan RTX 
    # output = output + str(2) + ','
    # # params share across different iteration, 0 false
    # output = output + str(0) + ','

    # # batch size
    # output = output + str(args.batch_size) + ","
    # # unroll factor
    # unroll_factor = args.unroll / args.num_timesteps
    # output = output + str(unroll_factor) + ","
    # # number of timesteps
    # output = output + str(args.num_timesteps) + ","

    # # type of sde 0 foward sde 1 foward-backward sde
    # output = output + str(args.num_timesteps) + ","
    y0 = jnp.ones((24, hidden_size))

    learning_rate = 1e-2
    learning_rate_fn = optax.exponential_decay(learning_rate, 1, 0.999)
    tx = optax.adam(learning_rate=learning_rate_fn)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

    start_time = time.time()

    for step in range(1000):
        rng, _ = jax.random.split(rng)
        # dW = jrandom.normal(rng, (args.num_timesteps, args.batch_size, args.dim))
        state, loss = train_step(step, state, y0, None)

        if step == 0:
            compile_time = time.time()
            iter_time = time.time()
        # print(f"iter: {time.time() - iter_time}")
        iter_time = time.time()
        # if step % 100 == 0 and step > 0:
        #     iter_time_list.append(time.time() - iter_time)
        #     iter_time = time.time()


    output = output + str(compile_time - start_time) + ','
    output = output + str(time.time() - compile_time)
    print(output)


train()

# @dataclass
# class Args:
#     batch_size: int
#     dim: int
#     num_timesteps: int
#     num_ts: int
#     num_iters: int
#     layers: Sequence[int]
#     unroll: int
#     T: float = 1.0


# def main():
    # train()


# if __name__ == '__main__':
#     # args = Args(batch_size=512, 
#     #                         dim=2,
#     #                         num_timesteps=1000,
#     #                         num_ts=10,
#     #                         num_iters=1000, 
#     #                         layers=[256, 256, dim], 
#     #                         unroll=1)
#     # main(args=args)
#     for batch_size in [24]:
#     # for batch_size in [512]:
#         for num_timesteps in [64]:
#         # for num_timesteps in [200, 250, 300]:
#             for layer in [256]:
#             # for layer in [128, 256, 512, 1024]:
#                 for layer_num in [3]:
#                 # for layer_num in [4, 5, 6]:
#                     for dim in [16]:
#                     # for dim in [3, 16, 32, 64]:
#                         layers = [layer] * layer_num + [dim]
#                         n = 16
#                         args = Args(batch_size=batch_size, 
#                                 dim=dim,
#                                 num_timesteps=num_timesteps,
#                                 num_ts=10,
#                                 num_iters=100, 
#                                 layers=layers, 
#                                 unroll=n)
#                         # while n <= 5:
#                         #     if n == 0:
#                         #         unroll = 1
#                         #     else:
#                         #         unroll = math.ceil(0.1 * n * num_timesteps)
#                         #         if unroll > 100:
#                         #             break
#                         #     args = Args(batch_size=batch_size, 
#                         #         dim=dim,
#                         #         num_timesteps=num_timesteps,
#                         #         num_ts=10,
#                         #         num_iters=1000, 
#                         #         layers=layers, 
#                         #         unroll=unroll)
#                         #     n += 1
#                         main(args=args)