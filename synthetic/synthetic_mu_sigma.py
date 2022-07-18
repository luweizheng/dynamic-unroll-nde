import time
from typing import Union
from dataclasses import dataclass
from functools import partial

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

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
            activation=lipswish,
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
        width_size,
        depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mf_key, sf_key = jrandom.split(key, 2)

        self.mf = MuField(hidden_size, width_size, depth, key=mf_key)
        self.sf = SigmaField(
            noise_size, hidden_size, width_size, depth, key=sf_key
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

class NeuralDE(eqx.Module):
    step: SDEStep
    noise_size: int
    hidden_size: int


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
        initial_key, step_key, readout_key = jrandom.split(key, 3)

        self.step = SDEStep(noise_size=noise_size, hidden_size=hidden_size, width_size=width_size, depth=depth, key=step_key)

        self.noise_size = noise_size
        self.hidden_size = hidden_size

    def make_cost_model_feature(self):

        def step_fn(carry, inp):
            return self.step(carry, inp)

        output = ""

        dummy_t0 = 0.0
        dummy_dt = 0.1

        dummy_y0 = jnp.ones((self.hidden_size, ))
        dummy_bm_key = jrandom.PRNGKey(0)
        carry = (0, dummy_t0, dummy_dt, dummy_y0, dummy_bm_key)
        hlo_module = jax.xla_computation(step_fn)(carry, None).as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        step_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
        step_bytes_access_gb = step_cost['bytes accessed'] / 1e9
        step_flops_g = step_cost['flops'] / 1e9
        
        # step bytes access in GB
        output = output + str(step_bytes_access_gb) + ','
        # step G FLOPS 
        output = output + str(step_flops_g) + ','
        # step Arithmetic Intensity
        output = output + str(step_flops_g / step_bytes_access_gb) + ','

        total_params = sum(p.size for p in jax.tree_leaves(eqx.filter(self.step, eqx.is_array)))

        output = output + str(total_params / 1e6) + ','

        return output

    def __call__(self, y0, num_timesteps, ts, key):
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = 1.0
        _, bm_key = jrandom.split(key, 2)

        def step_fn(carry, inp):
            return self.step(carry, inp)
        
        ys = solve(step_fn, y0, t0, dt0, num_timesteps, bm_key)
        
        return ys



def solve(step, y0, t0, dt, num_steps, bm_key):
    carry = (0, t0, dt, y0, bm_key)

    _, ys = jax.lax.scan(step, carry, xs=None, length=num_steps)

    return ys

@eqx.filter_jit
def train_step(step, model, y0, ts, num_timesteps, optimizer, opt_state, unroll, key):

    def loss_fn(model):

        # @partial(jit)
        # def forward_fn(params, y0, dW, ts, times):
        #     ys = train_state.apply_fn({'params': params}, y0, dW, ts, times)
        #     return ys

        # ys = forward_fn(params, y0, dW, ts, times)
        ys = jax.vmap(model, in_axes=[0, None, None, None])(y0, num_timesteps, ts, key)
        # dummy loss
        loss = jnp.sum(jnp.mean(ys, axis=0))

        return loss

    # @eqx.filter_value_and_grad
    # def grad_loss():
    #     return loss_fn(model)

    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, loss

def train():

    key = jrandom.PRNGKey(42)

    noise_size = 3
    hidden_size = 8
    width_size = 4
    depth = 5
    key = jrandom.PRNGKey(0)

    model = NeuralDE(
            noise_size,
            hidden_size,
            width_size,
            depth,
            key=key,
        )

    features = model.make_cost_model_feature()
    print(features)

    y0 = jnp.ones((24, hidden_size))
    num_timesteps = 64
    # ts = jnp.arange(start=0.0, stop=1.0, step=(1.0 - 0.0) / num_timesteps)
    ts = jnp.array([0.0, 1.0])

    learning_rate = 1e-2
    learning_rate_fn = optax.exponential_decay(learning_rate, 1, 0.999)
    optimizer = optax.adam(learning_rate=learning_rate_fn)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    start_time = time.time()

    for step in range(10):
        key, _ = jax.random.split(key)
        model, loss = train_step(step, model, y0, ts, num_timesteps, optimizer, opt_state, unroll=1, key=key)

        if step == 0:
            compile_time = time.time()
            iter_time = time.time()
        # print(f"iter: {time.time() - iter_time}")
        iter_time = time.time()
        # if step % 100 == 0 and step > 0:
        #     iter_time_list.append(time.time() - iter_time)
        #     iter_time = time.time()

    output = ""
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


# def main(args):
#     train(args)


# if __name__ == '__main__':
#     # args = Args(batch_size=512, 
#     #                         dim=2,
#     #                         num_timesteps=1000,
#     #                         num_ts=10,
#     #                         num_iters=1000, 
#     #                         layers=[256, 256, dim], 
#     #                         unroll=1)
#     # main(args=args)
#     for batch_size in [256]:
#     # for batch_size in [512]:
#         for num_timesteps in [200, 250, 300]:
#             for layer in [128, 256, 512, 1024]:
#                 for layer_num in [4, 5, 6]:
#                     for dim in [4, 16, 32, 64]:
#                         layers = [layer] * layer_num + [dim]
#                         n = 0
#                         while n <= 5:
#                             if n == 0:
#                                 unroll = 1
#                             else:
#                                 unroll = math.ceil(0.1 * n * num_timesteps)
#                                 if unroll > 100:
#                                     break
#                             args = Args(batch_size=batch_size, 
#                                 dim=dim,
#                                 num_timesteps=num_timesteps,
#                                 num_ts=10,
#                                 num_iters=1000, 
#                                 layers=layers, 
#                                 unroll=unroll)
#                             n += 1
#                             main(args=args)