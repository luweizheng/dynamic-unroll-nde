import time
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import equinox as eqx
from dataclasses import dataclass
from functools import partial

class MLP(eqx.Module):
    hidden_size: int
    depth:int
    width_size_list: list
    net: Sequence[eqx.nn.Linear]
    l_out: eqx.nn.Linear
    
    def __init__(self, hidden_size, width_size_list, depth, *, key):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.width_size_list = width_size_list
        in_key, out_key = jrandom.split(key, 2)
        l_in= eqx.nn.Linear(hidden_size, width_size_list[0], key=in_key)
        self.net = [l_in]
        wkeys = jrandom.split(key, depth - 1)
        for i, wkey in enumerate(wkeys):
            l = eqx.nn.Linear(width_size_list[i], width_size_list[i + 1], key=wkey)
            self.net.append(l)
        self.l_out = eqx.nn.Linear(width_size_list[depth - 1], hidden_size, key=out_key)

    def __call__(self, y):
        for l in self.net:
            y = l(y)
            y = jnn.relu(y)
        y = self.l_out(y)
        y = jnn.tanh(y)
        
        return y


class Unroll(eqx.Module):
    mlp: MLP
    unroll: int

    def __init__(self, hidden_size, width_size_list, depth, unroll=1, *, key):
        super(Unroll, self).__init__()
        self.mlp = MLP(hidden_size, width_size_list, depth, key=key)
        self.unroll = unroll
    
    def __call__(self, x, num_steps):
        def step_fn(x, input=None):
            y = self.mlp(x)

            return y, y
        
        x, _ = jax.lax.scan(step_fn, x, xs=None, length=num_steps, unroll=self.unroll)
        
        return x

@dataclass
class Args:
    batch_size: int

    # dim of SDE
    hidden_size: int
    num_steps: int
    num_iters: int
    
    # network
    depth: Sequence[int]
    width_size_list: list
    
    # dynamic unroll
    unroll: int
    seed:int = 5678
    
    #arch
    arch:str = 'Titan'
    
    
@eqx.filter_jit
def single_forward_fn(model, x, num_steps):
    y = model(x, num_steps)
    return y

@eqx.filter_jit
def batch_forward_fn(model, xs, num_steps):
    fn = partial(single_forward_fn, model)
    y = jax.vmap(fn, in_axes=(0, None))(xs, num_steps)
    loss = jnp.mean(y)
    return loss
    
def run(args):
    collection = []
    # collection.append(args.arch)
    # collection.append(args.unroll)
    key = jrandom.PRNGKey(args.seed)
    model = Unroll(args.hidden_size, args.width_size_list, args.depth, args.unroll, key=key)
    x = jnp.ones((args.batch_size, args.hidden_size))
    start_ts = time.time()
    for step in range(args.num_iters):
        batch_forward_fn(model, x, args.num_steps)
        if step == 0:
            compile_ts = time.time()
    
    collection.append(compile_ts - start_ts)
    collection.append(time.time() - compile_ts)
    print(','.join(map(str, collection)))


if __name__ == '__main__':
    args = Args(
        batch_size=64,
        hidden_size=16,
        num_steps=1000,
        num_iters=2000,
        depth=3,
        width_size_list=[64,64,64],
        unroll=1,
    )
    
    #warm up 
    run(args)
    
    unroll_list = [1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 100]
    for unroll in unroll_list:
        args.unroll = unroll
        
        run(args)
        