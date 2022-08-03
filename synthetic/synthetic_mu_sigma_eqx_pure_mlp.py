from email import iterators
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
import itertools
# We use GPU as the default backend.
# If you want to use cpu as backend, uncomment the following line.
# config.update("jax_platform_name", "cpu")

def lipswish(x):
    return 0.909 * jnn.silu(x)


class MuField(eqx.Module):
    l_out: eqx.nn.Linear
    net: Sequence[eqx.nn.Linear]
    hidden_size:int
    depth:int
    
    
    def __init__(self, hidden_size, width_size_list, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.depth = depth
        in_key, out_key = jrandom.split(key, 2)
        l_in= eqx.nn.Linear(hidden_size+1, width_size_list[0], key=in_key)
        self.net = [l_in]
        wkeys = jrandom.split(key, depth-1)
        for i, wkey in enumerate(wkeys):
            l = eqx.nn.Linear(width_size_list[i], width_size_list[i+1], key=wkey)
            self.net.append(l)
        self.l_out = eqx.nn.Linear(width_size_list[depth-1], hidden_size, key=out_key)
        
    def __call__(self, t, y):
        y = jnp.concatenate([t, y])
        for l in self.net:
            y = l(y)
            y = jnn.relu(y)
        y = self.l_out(y)
        y = jnn.tanh(y)
        
        return y
            
        
        


class SigmaField(eqx.Module):
    l_out: eqx.nn.Linear
    net: Sequence[eqx.nn.Linear]    
    noise_size: int
    hidden_size: int

    def __init__(
        self, noise_size, hidden_size, width_size_list, depth, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        self.noise_size = noise_size
        self.hidden_size = hidden_size
        in_key, out_key = jrandom.split(key, 2)
        l_in= eqx.nn.Linear(hidden_size+1, width_size_list[0], key=in_key)
        self.net = [l_in]
        wkeys = jrandom.split(key, depth-1)
        for i, wkey in enumerate(wkeys):
            l = eqx.nn.Linear(width_size_list[i], width_size_list[i+1], key=wkey)
            self.net.append(l)
        self.l_out = eqx.nn.Linear(width_size_list[depth-1], hidden_size * noise_size, key=out_key)

    def __call__(self, t, y):
        y = jnp.concatenate([t,y])
        for l in self.net:
            y = l(y)
            y = lipswish(y)
        y = self.l_out(y)
        y = jnn.tanh(y)
        y = y.reshape(self.hidden_size, self.noise_size)
        return y

class SDEStep(eqx.Module):
    mf: MuField  # drift
    sf: SigmaField  # diffusion
    noise_size: int

    def __init__(
        self,
        noise_size,
        hidden_size,
        width_size_list,
        depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mf_key, sf_key = jrandom.split(key, 2)

        self.mf = MuField(hidden_size, width_size_list, depth, key=mf_key)
        self.sf = SigmaField(
            noise_size, hidden_size, width_size_list, depth, key=sf_key
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
    depth: int
    width_size_list: list

    def __init__(
        self,
        noise_size,
        hidden_size,
        width_size_list,
        depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        step_key, _ = jrandom.split(key, 2)

        self.step = SDEStep(noise_size=noise_size, hidden_size=hidden_size, width_size_list=width_size_list, depth=depth, key=step_key)

        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.width_size_list = width_size_list
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
        
        # count of width
        w32=0
        w64=0
        w128=0
        w256=0
        for w in self.width_size_list:
            if w == 32:
                w32 += 1
            elif w == 64:
                w64 += 1
            elif w == 128:
                w128 += 1
            elif w == 256:
                 w256 += 1
        
        features.append(w32)
        features.append(w64)
        features.append(w128)
        features.append(w256)

        return features

    def __call__(self, y0, t0, dt, num_timesteps, unroll, key):

        _, bm_key = jrandom.split(key, 2)

        def step_fn(carry, inp):
            return self.step(carry, inp)

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
            args.width_size_list,
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
    print(','.join(map(str, features)))
    
    del model

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
    width_size_list: list
    
    # dynamic unroll
    unroll: int
    T: float = 1.0


def main():
    
    # warm up run
    args = Args(batch_size=128, 
            hidden_size=16,
            noise_size=16,
            num_timesteps=50,
            num_iters=1000, 
            depth=4, 
            width_size_list=[64,64,64,64],
            unroll=1)
    # dummy run
    train(args)
    
    width_sizes = [32] + [64] + [128] + [256]

    for batch_size in [64, 128, 256]:
        for num_timesteps in [50, 100, 200]:
            for depth in [6]:
                width_size_lists = [list(perm) for perm in itertools.combinations_with_replacement(width_sizes, depth) if sorted(perm) == list(perm)]
                for hidden_size in [16, 32, 64]:
                    for width_size_list in width_size_lists:
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
                                width_size_list=width_size_list,
                                unroll=unroll)
                            n += 1
                            train(args=args)


if __name__ == '__main__':
    main()