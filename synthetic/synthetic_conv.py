import functools as ft
from dataclasses import dataclass
from typing import Sequence
import time
from diffrax.misc import ω
import einops  # https://github.com/arogozhnikov/einops
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import math
import equinox as eqx
import itertools

class MLP(eqx.Module):
    input_size:int
    output_size:int
    depth:int
    width_size_list:list
    l_out: eqx.nn.Linear
    net:Sequence[eqx.nn.Linear]
    
    def __init__(self, input_size, output_size, width_size_list, depth, *, key) :
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        in_key, out_key = jr.split(key, 2)
        l_in = eqx.nn.Linear(input_size, width_size_list[0], key=in_key)
        self.net = [l_in]
        wkeys = jr.split(key, depth - 1)
        for i, wkey in enumerate(wkeys):
            l = eqx.nn.Linear(width_size_list[i], width_size_list[i+1], key=wkey)
            self.net.append(l)
        
        self.l_out = eqx.nn.Linear(width_size_list[depth - 1], output_size, key=out_key)
        
    
    def __call__(self, y):
        for l in self.net:
            y = l(y)
            y = jnn.relu(y)
        y = self.l_out(y)
        return y
            
        

class MixerBlock(eqx.Module):
    patch_mixer: MLP
    hidden_mixer: MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self, num_patches, hidden_size, width_size_list, depth,*, key
    ):
        tkey, ckey = jr.split(key, 2)
        self.patch_mixer = MLP(
            num_patches, num_patches, width_size_list, depth=depth, key=tkey
        )
        self.hidden_mixer = MLP(
            hidden_size, hidden_size, width_size_list, depth=depth, key=ckey
        )
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))

    def __call__(self, y):
        y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
        y = einops.rearrange(y, "c p -> p c")
        y = y + jax.vmap(self.hidden_mixer)(self.norm2(y))
        y = einops.rearrange(y, "p c -> c p")
        return y


class Mixer2d(eqx.Module):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.ConvTranspose2d
    blocks: list
    norm: eqx.nn.LayerNorm
    t1: float

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        width_size_list,
        num_blocks,
        block_depth,
        t1,
        *,
        key,
    ):
        
        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        self.conv_in = eqx.nn.Conv2d(
            input_size + 1, hidden_size, patch_size, stride=patch_size, key=inkey
        )
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, input_size, patch_size, stride=patch_size, key=outkey
        )
        self.blocks = [
            MixerBlock(
                num_patches, hidden_size, width_size_list, block_depth,  key=bkey
            )
            for bkey in bkeys
        ]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.t1 = t1
    
    
    def make_cost_model_feature(self):

        def step_fn(carry, inp):
            return self.step(carry, inp)

        dummy_t0 = 0.0
        dummy_dt = 0.1

        dummy_y0 = jnp.ones((self.hidden_size, ))
        dummy_bm_key = jr.PRNGKey(0)
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

    def __call__(self, t, y):
        t = t / self.t1
        _, height, width = y.shape
        t = einops.repeat(t, "-> 1 h w", h=height, w=width)
        y = jnp.concatenate([y, t])
        y = self.conv_in(y)
        _, patch_height, patch_width = y.shape
        y = einops.rearrange(y, "c h w -> c (h w)")
        for block in self.blocks:
            y = block(y)
        y = self.norm(y)
        y = einops.rearrange(y, "c (h w) -> c h w", h=patch_height, w=patch_width)
        return self.conv_out(y)

def single_loss_fn(model, weight, int_beta, data, t, key):
    mean = data * jnp.exp(-0.5 * int_beta(t))
    var = jnp.maximum(1 - jnp.exp(-int_beta(t)), 1e-5)
    std = jnp.sqrt(var)
    noise = jr.normal(key, data.shape)
    y = mean + std * noise
    pred = model(t, y)
    return weight(t) * jnp.mean((pred + noise / std) ** 2)


def batch_loss_fn(model, weight, int_beta, data, t1, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, model, weight, int_beta)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(data, t, losskey))


@eqx.filter_jit
def single_sample_fn(model, int_beta, data_shape, dt0, t1, unroll, key):
    def drift(t, y, args):
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(t, y))
    
    def step_fn_euler(carry, input=None):
        (i, t0, y0, dt) = carry
        t1 = t0 + dt
        control = dt
        vf = drift(t0, y0, args=None)
        vf_prod = jax.tree_map(lambda v: control * v, vf)
        y1 = (y0**ω + vf_prod** ω).ω

        carry = (i+1, t1, y1, dt)
        return carry, y1
        
    def solve(step, t0, y0, dt, num_timesteps):
        
        carry = (0, t0, y0, dt)
        
        _, ys = jax.lax.scan(step, carry, xs=None, length=num_timesteps, unroll=unroll)
        
        return ys
        
    t0 = 0.0
    y1 = jr.normal(key, data_shape)
    
    t0 = jnp.asarray(t0, dtype=jnp.float32)
    t1 = jnp.asarray(t1, dtype=jnp.float32)
    dt0 = jnp.asarray(dt0, dtype=jnp.float32)
    ys = solve(step_fn_euler, t1, y1, -dt0, num_timesteps=100)
    # reverse time, solve from t1 to t0

    return ys[-1]


@eqx.filter_jit
def make_step(model, weight, int_beta, data, t1, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, weight, int_beta, data, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state


def train(args):
    key = jr.PRNGKey(42)
    y0 = jnp.ones((args.batch_size, args.height, args.width))
    model = Mixer2d(y0.shape, args.patch_size, args.hidden_size, args.width_size_list, args.num_blocks, args.block_depth ,args.t1, key=key)
    
    int_beta = lambda t: t  # Try experimenting with other options here!
    weight = lambda t: 1 - jnp.exp(
        -int_beta(t)
    )  # Just chosen to upweight the region near t=0.
    lr=3e-4
    opt = optax.adabelief(lr)
    # Optax will update the floating-point JAX arrays in the model.
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    
    features = model.make_cost_model_feature()
    features.append(args.batch_size)
    features.append(args.num_timesteps)
    features.append(args.unroll)
    
    start_time = time.time()
    
    for step in range(1000):
        value, model, train_key, opt_state = make_step(
            model, weight, int_beta, y0, args.t1, train_key, opt_state, opt.update
        )

        if step == 0:
            compile_time = time.time()
    
    features.append(compile_time - start_time)
    features.append(time.time() - compile_time)
    print(','.join(map(str, features)))
    
    del model
        
@dataclass
class Args:
    batch_size: int

    # dim of SDE
    hidden_size: int
    patch_size:int
    height:int
    width:int
    
    
    num_timesteps: int
    num_iters: int
    
    # network
    block_depth: int
    width_size_list: list
    
    # dynamic unroll
    unroll: int
    t1: float = 1.0
    

def main(args):
    train(args)

if __name__ == '__main__':
    # test code
    args = Args(
            batch_size=128, 
            hidden_size=16,
            patch_size=4,
            height=16,
            width=16,
            num_timesteps=50,
            num_iters=1000, 
            block_depth=3, 
            width_size_list=[64]*3,
            unroll=1)
    # warm up run
    main(args=args)
    
    width_sizes = [32] + [64] + [128]
    
    for batch_size in [64, 128, 256]:
        for num_timesteps in [50, 100, 200]:
            for hidden_size in [16, 32, 64]:
                for width in [16, 32, 64]:
                    for height in [16, 32, 64]:
                        for patch_size in [2, 4, 8]:
                            for block_depth in [1, 2, 3]:
                                width_size_lists = [list(perm) for perm in itertools.combinations_with_replacement(width_sizes, block_depth) if sorted(perm) == list(perm)]
                                for width_size_list in width_size_lists:
                                    n = 0
                                    while n <= 5:
                                        if n == 0:
                                            unroll = 1
                                        else:
                                            unroll = math.ceil(0.1 * n * num_timesteps)
                                            if unroll > 100:
                                                break
                                        args = Args(
                                            batch_size=batch_size, 
                                            hidden_size=hidden_size,
                                            patch_size=patch_size,
                                            height=height,
                                            width=width,
                                            num_timesteps=num_timesteps,
                                            num_iters=1000, 
                                            block_depth=block_depth, 
                                            width_size_list=width_size_list,
                                            unroll=unroll)
                                        n += 1
                                        main(args=args)
                            
                        


