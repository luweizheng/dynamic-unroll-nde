from typing import Sequence, Union
import jax.numpy as jnp
import numpy  as np
import jax.nn as jnn
import jax.random as jr
import jax
import optax
import equinox as eqx
import os
import argparse
import logging
import math
import time
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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
        out = x
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
    integration_time: jnp.array
    t0:float
    dt:float
    num_timesteps:int
    def __init__(self, odefunc):
        self.odefunc = odefunc
        self.integration_time = jnp.asarray([0.0, 1.0])
        self.t0 = 0.0
        self.dt = 0.01
        self.num_timesteps=100 
        
    def step(self, carry, input=None):
        (i, t0, dt, y0) = carry 
        t = t0 + i * dt
        # dy/dt = drift(t,y)
        
        dy = dt * self.odefunc(t, y0)
        y1 = y0 + dy
        carry = (i+1, t0, dt, y1)
        return carry, y1

    
    def __call__(self, y0, unroll=1):
        ys = solve(self.step, y0, self.t0, self.dt, num_timesteps=self.num_timesteps, unroll=unroll)
        return ys


class Flatten(eqx.Module):

    def __call__(self, x):
        # x = jnp.squeeze(x, axis=1)
        return x.reshape(-1)


class ODENet(eqx.Module):
    downsampling_layers:list
    feature_layers: Sequence[ODEBlock]
    fc_layers:list 
    def __init__(self, key) -> None:
        super().__init__()
        keys = jr.split(key, 5)
        self.downsampling_layers = [
            #32 224
            eqx.nn.Conv2d(3, 64, 3, 1, key=keys[0]),
            eqx.nn.GroupNorm(32, 64),
            jnn.relu,
            eqx.nn.Conv2d(64, 64, 4, 2, 1, key=keys[1]),
            eqx.nn.GroupNorm(32, 64),
            jnn.relu,
            eqx.nn.Conv2d(64, 64, 4, 2, 1, key=keys[2]),
        ]
        self.feature_layers = [ODEBlock(ODEfunc(64, key=keys[3]))] 
        self.fc_layers = [eqx.nn.GroupNorm(32, 64), jnn.relu, eqx.nn.AvgPool2D((6, 6), stride=1), Flatten(), eqx.nn.Linear(256, 10, key=keys[4])]
    def __call__(self, x, unroll):
        for dl in self.downsampling_layers:
            x = dl(x)
        for fl in self.feature_layers:
            x = fl(x, unroll=unroll)
            
        for fc in self.fc_layers:
            x = fc(x)
        # x = x.T
        return x

@eqx.filter_jit
def single_loss_fn(model, unroll, image, label):
    logits = model(image, unroll)
    loss = optax.softmax_cross_entropy(logits=logits, labels=label)
    return loss

@eqx.filter_value_and_grad
def batch_loss_fn(model, images, labels, unroll):
    loss_fn = ft.partial(single_loss_fn, model, unroll)
    loss_fn = jax.vmap(loss_fn, in_axes=[0,0])
    # logits = jax.vmap(model, in_axes=0)(images)
    labels = jnn.one_hot(labels, 1)
    loss = jnp.mean(loss_fn(images, labels))
    return loss
    

def get_cifar10_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=32, drop_last=True
    )


    train_eval_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=16, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=False, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=16, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = jnp.asarray(x, dtype=jnp.float32)
        y = jnp.asarray(y)
        y = jnn.one_hot(y, 10)
        logits = jax.vmap(model, in_axes=(0, None))(x, 10)
        target_class = jnp.argmax(y, axis=1)
        # logits = jnp.squeeze(logits, axis=1)
        predicted_class = jnp.argmax(logits, axis=1)
        total_correct += jnp.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


@eqx.filter_jit
def train_step(model, x, y, opt, opt_state, unroll):
    _ , grads = batch_loss_fn(model, x, y, unroll=unroll)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model

def train(args):
    print(f"unroll: {args.unroll}")
    train_loader, test_loader, train_eval_loader = get_cifar10_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )
    
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    key = jax.random.PRNGKey(0)
    lr = 0.1
    opt = optax.sgd(lr, momentum=0.9)
    model = ODENet(key=key)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    for itr in range(args.num_epochs * batches_per_epoch):
        x, y = data_gen.__next__()
        x = jnp.asarray(x, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        if itr == 0:
            start_time = time.time()
            iter_time = start_time
        model = train_step(model, x, y, opt, opt_state, args.unroll)
        if itr == 0:
            # compile at the first iteration
            compile_time = time.time() - start_time
            iter_time = time.time()
            print(f"compile_time: {compile_time:.4f}")
        if itr != 0 and (itr % batches_per_epoch == 0 or itr == (args.num_epochs * batches_per_epoch - 1)):
            # train_acc = accuracy(model, train_eval_loader)
            # val_acc = accuracy(model, test_loader)
            epoch_time = time.time() - iter_time
            iter_time = time.time()
            print(
                "Epoch {:04d} | "
                "time {:.4f}".format(
                    math.ceil(itr / batches_per_epoch), epoch_time
                )
            )
        if itr == (args.num_epochs * batches_per_epoch - 1):
            print(f"total time: {time.time() - start_time}")


@dataclass
class Args:
    batch_size: int
    num_epochs: int
    test_batch_size: int
    data_aug: bool = False
    unroll: int = 4

if __name__ == '__main__':
    args = Args(batch_size=256, num_epochs=10, test_batch_size=500)
    train(args)