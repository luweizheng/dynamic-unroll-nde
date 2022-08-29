from typing import Sequence, Union
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import jax
import optax
import equinox as eqx
import os
import time
from dataclasses import dataclass
import functools as ft

os.environ['NO_GCE_CHECK'] = 'true'

def norm(dim):
    return eqx.nn.GroupNorm(min(32, dim), dim)

class ConcatConv2d(eqx.Module):
    _layer: Union[eqx.nn.Conv2d, eqx.nn.ConvTranspose2d]
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, transpose=False, *, key):
        super(ConcatConv2d, self).__init__()
        module = eqx.nn.ConvTranspose2d if transpose else eqx.nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, key=key
        )   

    def __call__(self, t, x):
        tt = jnp.ones_like(x[1, :, :]) * t
        tt = jnp.expand_dims(tt, axis=0)
        ttx = jnp.concatenate([tt, x], 0)
        return self._layer(ttx)



class ODEfunc(eqx.Module):
    norm1: eqx.nn.GroupNorm
    relu: jnn.relu
    conv1: ConcatConv2d
    norm2:eqx.nn.GroupNorm
    conv2: ConcatConv2d
    norm3:eqx.nn.GroupNorm
    def __init__(self, dim, *, key):
        keys = jr.split(key, 2)
        self.norm1 = eqx.nn.GroupNorm(32, dim)
        self.relu = jnn.relu
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1, key=keys[0])
        self.norm2 = eqx.nn.GroupNorm(min(32, dim), dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1, key=keys[1])
        self.norm3 = eqx.nn.GroupNorm(32, dim)

    def __call__(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out 


def solve(step, y0, t0, dt, num_timesteps, unroll=1):
    carry = (0, t0, dt, y0)

    _, ys = jax.lax.scan(step, carry, xs=None, length=num_timesteps, unroll=unroll)

    return ys[-1]


class ODEBlock(eqx.Module):
    odefunc: ODEfunc
    t0:float
    dt:float
    num_timesteps:int
    unroll:int
    def __init__(self, odefunc, num_timesteps=100, unroll=1):
        self.odefunc = odefunc
        self.t0 = 0.0
        self.dt = 0.01
        self.num_timesteps=num_timesteps
        self.unroll = unroll
    def step(self, carry, input=None):
        (i, t0, dt, y0) = carry 
        t = t0 + i * dt

        dy = dt * self.odefunc(t, y0)
        y1 = y0 + dy
        carry = (i+1, t0, dt, y1)
        return carry, y1


    def __call__(self, y0):
        ys = solve(self.step, y0, self.t0, self.dt, num_timesteps=self.num_timesteps, unroll=self.unroll)
        return ys


class Flatten(eqx.Module):

    def __call__(self, x):
        return x.reshape(-1)


class ODENet(eqx.Module):
    downsampling_layers:list
    feature_layers: Sequence[ODEBlock]
    fc_layers:list 
    def __init__(self, num_timesteps=100, unroll=1, *,key) -> None:
        super().__init__()
        keys = jr.split(key, 5)
        self.downsampling_layers = [
            eqx.nn.Conv2d(3, 64, 3, 1, key=keys[0]),
            eqx.nn.GroupNorm(32, 64),
            jnn.relu,
            eqx.nn.Conv2d(64, 64, 4, 2, 1, key=keys[1]),
            eqx.nn.GroupNorm(32, 64),
            jnn.relu,
            eqx.nn.Conv2d(64, 64, 4, 2, 1, key=keys[2]),
        ]
        self.feature_layers = [ODEBlock(ODEfunc(64, key=keys[3]), num_timesteps=num_timesteps, unroll=unroll)] 
        self.fc_layers = [eqx.nn.GroupNorm(32, 64), jnn.relu, eqx.nn.AvgPool2D((27, 27), stride=1), Flatten(), eqx.nn.Linear(64, 10, key=keys[4])]
    def __call__(self, x):
        for dl in self.downsampling_layers:
            x = dl(x)
        for fl in self.feature_layers:
            x = fl(x)
        for fc in self.fc_layers:
            x = fc(x)
        return x

@eqx.filter_jit
def single_loss_fn(model, image, label):
    logits = model(image)
    loss = optax.softmax_cross_entropy(logits=logits, labels=label)
    return loss

@eqx.filter_value_and_grad
def batch_loss_fn(model, images, labels):
    loss_fn = ft.partial(single_loss_fn, model)
    loss_fn = jax.vmap(loss_fn, in_axes=[0,0])
    labels = jnn.one_hot(labels, 10)
    loss = jnp.mean(loss_fn(images, labels))
    return loss

@eqx.filter_jit
def acc_step(model, x, y):
    logits = jax.vmap(model, in_axes=(0))(x)
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(logits, axis=1)
    correct = jnp.sum(predicted_class == target_class)
    return correct


@eqx.filter_jit
def train_step(model, x, y, opt, opt_state):
    _ , grads = batch_loss_fn(model, x, y)

    return model, opt_state

def train(args):
    collection = []
    # collection.append(args.arch)
    # collection.append(args.unroll)
    key = jax.random.PRNGKey(0)
    lr = 0.1
    opt = optax.sgd(lr, momentum=0.9)
    model = ODENet(num_timesteps=args.num_timesteps ,key=key)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    x = jnp.ones((64, 3, 112, 112), dtype=jnp.float32)
    dummy_y = [0,1,2,3,4,5,6,7,8,9]*6+[0,1,2,3]
    y = jnp.asarray(dummy_y, dtype=jnp.float32)
    start_time = time.time()
    for itr in range(args.num_iters):
        model, opt_state = train_step(model, x, y, opt, opt_state)
        if itr == 0:
            compile_time = time.time()
    collection.append(args.unroll)
    collection.append(compile_time - start_time)
    collection.append(time.time() - compile_time)
    print(','.join(map(str, collection)))

@dataclass
class Args:
    batch_size: int
    num_timesteps: int
    num_iters: int
    unroll: int
    data_aug: bool = False
    arch: str = 'Titan'
     

if __name__ == '__main__':
    args = Args(batch_size=128, 
                num_timesteps=2000, 
                num_iters=200000, 
                unroll=1)
    
    # warm up
    train(args)
    
    unroll_list = [1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 100]
    for unroll in unroll_list:
        args.unroll = unroll
        train(args) 