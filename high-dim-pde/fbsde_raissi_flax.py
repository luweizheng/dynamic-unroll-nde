"""Methods to train Neural Forward-Backward Stochastic Differential Equation."""
import argparse
import math
from dataclasses import dataclass
import time
from functools import partial

from typing import Any, Callable, Sequence, Tuple


import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from flax.core import broadcast
from flax import linen as nn
from flax.training.train_state import TrainState
from flax import struct
from flax import core

import optax
from flax.core.frozen_dict import freeze

Array = jnp.ndarray
ModuleDef = Any
EquProblemDef = Any


@jax.jit
def sum_square_error(y, y_pred):
    """Computes the sum of square error."""
    return jnp.sum(jnp.square(y - y_pred))

class FBSDEProblem(struct.PyTreeNode):
    """Base class for a Forward-Backward SDE Problem.
    """
    g_fn: Callable = struct.field(pytree_node=False)
    dg_fn: Callable = struct.field(pytree_node=False)
    mu_fn: Callable = struct.field(pytree_node=False)
    sigma_fn: Callable = struct.field(pytree_node=False)
    phi_fn: Callable = struct.field(pytree_node=False)
    x0: Array
    tspan: tuple[float, float]
    num_timesteps: int = struct.field(pytree_node=False)
    dim: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, *, g_fn, mu_fn, sigma_fn, phi_fn, x0, tspan, num_timesteps, dim, **kwargs):
        """Creates a new instance with input parameters."""
        
        def dg_fn(X):
            y, vjp_func = jax.vjp(g_fn, X)
            return vjp_func(jnp.ones(y.shape))[0]

        return cls(
            g_fn=g_fn,
            dg_fn=dg_fn,
            mu_fn=mu_fn,
            sigma_fn=sigma_fn,
            phi_fn=phi_fn,
            x0=x0,
            tspan=tspan,
            num_timesteps=num_timesteps,
            dim=dim,
            **kwargs,
        )

class FBSDEModel(struct.PyTreeNode):
    step: int
    u_net_apply_fn: Callable = struct.field(pytree_node=False)
    # u_params: core.FrozenDict[str, Any]
    fbsde_net_apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    # initial_carry: Tuple
    equ_problem: EquProblemDef
    batch_size: int                 # batch size or number of trajectories
    num_timesteps: int              # number of timesteps
    dim: int                        # dimension of assets
    batch_stats: core.FrozenDict[str, Any] = None

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
            grads: Gradients that have the same pytree structure as `.params`.
            **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
            An updated instance of `self` with `step` incremented by one, `params`
            and `opt_state` updated by applying `grads`, and additional attributes
            replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )
    
    
    @classmethod
    def create(cls, *, mdl, layers, equ_problem, batch_size, num_timesteps, dim, tx, rng=jrandom.PRNGKey(42), **kwargs):
        """Creates a new model instance with parameters."""
        
        dt = (equ_problem.tspan[1] - equ_problem.tspan[0]) / equ_problem.num_timesteps
        
        class UNet(nn.Module):
            u: ModuleDef
            layers: Sequence

            @nn.compact
            def __call__(self, t, x):
                u = self.u(layers)
                (y, bwd) = nn.vjp(lambda mdl, x: mdl(t, x), u, x)
                dudx = bwd(jnp.ones(y.shape))
                return y, dudx[0]

        class FBSDECell(nn.Module):
            u_net: Callable[..., nn.Module]
            dt: float

            @nn.compact
            def __call__(self, carry, t, dW):
                # `t` and `W` are (batch_size, num_timestep, dim)
                # it have input data across iterations 
                i, x0, y0, z0 = carry

                # use `i` to index input data
                t0 = i * t
                t1 = (i + 1) * t
                # t0 = t[:, i-1, :]
                # t1 = t[:, i, :]
                # W0 = W[:, i-1, :]
                # W1 = W[:, i, :]

                x1 = x0 + equ_problem.mu_fn(t0, x0, y0, z0) * self.dt + \
                        equ_problem.sigma_fn(t0, x0, y0) * dW
                y1_tilde = y0 + equ_problem.phi_fn(t0, x0, y0, z0) * self.dt + \
                    jnp.sum(z0 * equ_problem.sigma_fn(t0, x0, y0) * dW, axis=1, keepdims=True)
                
                y1, z1 = self.u_net(t1, x1)

                carry = (i+1, x1, y1, z1)
                outputs = (x1, y1_tilde, y1)
                return carry, outputs


        class FBSDE(nn.Module):
            # u_net: ModuleDef
            # equ_problem: Any

            @nn.compact
            def __call__(self, carry, t, W, unroll=1):
                
                fbsdes = nn.scan(FBSDECell,
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=(nn.broadcast, 1),
                        out_axes=0,
                        unroll=unroll)
                y = fbsdes(name="fbsde", 
                    u_net=u_net,
                    dt=dt)(carry, t, W)
                return y

        output = ""
        
        u_net = UNet(u=mdl, layers=layers)
        fbsde_net = FBSDE()

        (y0, z0), u_du_variables = u_net.init_with_output(
            rng, t=jnp.zeros([batch_size, 1]), x=equ_problem.x0)

        initial_carry = (0, equ_problem.x0, y0, z0)

        cell = FBSDECell(u_net, dt)
        cell_variable = cell.init(rng, initial_carry, t=jnp.zeros([batch_size, 1]), dW=jnp.zeros([batch_size, equ_problem.dim]))

        # hlo_module = jax.xla_computation(cell.apply)({'params': cell_variable['params']}, initial_carry, t=jnp.zeros([batch_size, 1]), dW=jnp.zeros([batch_size, equ_problem.dim])).as_hlo_module()
        hlo_module = jax.xla_computation(u_net.apply)({'params': u_du_variables['params']}, t=jnp.zeros([batch_size, 1]), x=jnp.zeros([batch_size, equ_problem.dim])).as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
        bytes_access_gb = cost['bytes accessed'] / 1e9
        flops_g = cost['flops'] / 1e9
        output = output + str(bytes_access_gb) + ','
        output = output + str(flops_g) + ','
        output = output + str(flops_g / bytes_access_gb) + ','
        output = output + str(len(layers)) + ','
        # params share across different iteration, 0 false
        output = output + str(0) + ','
        # print(f"bytes_access: {bytes_access_gb}")
        # print(f"flops: {flops_g}")

        total_params = sum(p.size for p in jax.tree_leaves(cell_variable['params']))
        
        # million params
        output = output + str(total_params / 1e6) + ','
        # print(f"total_params: {total_params / 1e9}")

        fbsde_params = {
            'fbsde':  {
                'u_net': u_du_variables['params']
            }
        }
        fbsde_params = freeze(fbsde_params)

        opt_state = tx.init(fbsde_params)

        return cls(
            step=0,
            u_net_apply_fn=u_net.apply,
            fbsde_net_apply_fn=fbsde_net.apply,
            params=fbsde_params,
            tx=tx,
            opt_state=opt_state,
            equ_problem=equ_problem,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            dim=dim,
            **kwargs,
        ), output
    

def fetch_minibatch(model, rng):  # Generate time + a Brownian motion
    T = model.equ_problem.tspan[1]
    M = model.batch_size
    N = model.num_timesteps
    D = model.dim

    dt = T / N * jnp.ones((M, 1))
    dW = jnp.sqrt(T / N) * jrandom.normal(rng, shape=(M, N, D))

    return dt, dW

@partial(jax.jit, static_argnums=2)
def train_step(model, data, unroll=1):
    # batch_size = model.batch_size
    t, W = data

    def loss_fn(params):
        loss = 0.0
        
        (y0, z0) = model.u_net_apply_fn(
            {'params': params['fbsde']['u_net']}, 
            t=t * 0,
            x=model.equ_problem.x0)
        # define initial carry for nn.scan
        x0 = model.equ_problem.x0
        initial_carry = (0, x0, y0, z0)

        out_carry, out_val = model.fbsde_net_apply_fn(
            {'params': params}, 
            carry=initial_carry, 
            t=t, W=W,
            unroll=unroll)

        (_, x_final, y_final, z_final) = out_carry
        (x, y_tilde_list, y_list) = out_val

        loss += sum_square_error(y_tilde_list, y_list)
        loss += sum_square_error(y_final, model.equ_problem.g_fn(x_final))
        loss += sum_square_error(z_final, model.equ_problem.dg_fn(x_final))

        return (loss, y_list)

    (loss, y), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(model.params)
    
    model = model.apply_gradients(grads=grads)

    return model, loss, y

def train(model, num_iters, output, batch_size, unroll=1, rng=jrandom.PRNGKey(42), verbose=True):
    start_time = time.time()
    iter_time_list = []
    
    for i in range(num_iters):
        rng, _ = jax.random.split(rng)
        data = fetch_minibatch(model, rng)
        model, loss, y_pred = train_step(model, data, unroll)

        if verbose:
            if i == 0:
                compile_time = time.time()
                iter_time = time.time()
            # if i % 100 == 0 and i > 0:
            #     iter_time_list.append(time.time() - iter_time)
            #     iter_time = time.time()

    running_time = (compile_time - start_time, time.time() - compile_time)
    
    return model, running_time


def main(args):
    M = args.batch_size  # number of trajectories (batch size)
    N = args.num_timesteps  # number of time snapshots
    D = args.dim  # number of dimensions
    T = 1.0 # expire time

    def g_fn(X):
        return jnp.sum(X ** 2, axis=-1, keepdims=True)

    def mu_fn(t, X, Y, Z):
        del t, Y, Z
        return jnp.zeros_like(X)

    def sigma_fn(t, X, Y):
        del t, Y
        return 0.4 * X

    def phi_fn(t, X, Y, Z):
        del t
        return 0.05 * (Y - jnp.sum(X * Z, axis=1, keepdims=True))

    x0 = jnp.array([[1.0, 0.5] * int(args.dim / 2)])
    x0 = jnp.broadcast_to(x0, (args.batch_size, args.dim))

    tspan = (0.0, 1.0)
    num_timesteps = 50

    bsb_problem = FBSDEProblem.create(g_fn=g_fn, 
        mu_fn=mu_fn, sigma_fn=sigma_fn, 
        phi_fn=phi_fn, x0=x0, 
        tspan=tspan, 
        num_timesteps=num_timesteps,
        dim=D)

    class FNN(nn.Module):
        layers: Sequence

        @nn.compact
        def __call__(self, t, x, train: bool = True):
            x = jnp.hstack((t, x))
            for i in range(len(self.layers)):
                x = nn.Dense(features=self.layers[i])(x)
                x = nn.relu(x)
            x = nn.Dense(features=1)(x)
            return x

    learning_rate = 1e-3
    tx = optax.adam(learning_rate=learning_rate)

    fbsdeModel, output = FBSDEModel.create(mdl=FNN, layers=args.layers, equ_problem=bsb_problem, batch_size=M, num_timesteps=N, dim=D, tx=tx)

    output = output + str(args.batch_size) + ","
    output = output + str(args.unroll) + ","
    output = output + str(args.num_timesteps) + ","

    (y0, z0) = fbsdeModel.u_net_apply_fn({'params': fbsdeModel.params['fbsde']['u_net']}, t=jnp.zeros([M, 1]), x=x0)

    num_iters = args.num_iters
    # M = jnp.array(M, dtype=jnp.int32)
    
    fbsdeModel, running_time = train(fbsdeModel, num_iters, output, batch_size = M, unroll=args.unroll)
    print(f"unroll: {args.unroll}, actuall time {running_time[0] + running_time[1] * 10}")

@dataclass
class Args:
    batch_size: int
    num_timesteps: int
    dim: int
    num_iters: int
    layers: Sequence[int]
    unroll: int

if __name__ == '__main__':
    # AP = argparse.ArgumentParser()
    # AP.add_argument('--batch_size', type=int, default=30,
    #                 help='')
    # AP.add_argument('--unroll', type=int, default='1',
    #                 help='')
    # AP.add_argument('--num_timesteps', type=int, default='100',
    #                 help='')
    # AP.add_argument('--num_iters', type=int, default=1000, 
    #                 help='')
    # AP.add_argument('--layers', nargs='+', type=int, default=[256])

    # args = AP.parse_args()

    args = Args(batch_size=128, 
            num_timesteps=50,
            dim=16,
            num_iters=1000, 
            layers=[64, 64, 1], 
            unroll=10)
    unroll_list = [2, 5, 10, 15, 20, 30, 40, 50]
    for unroll in unroll_list:
        args.unroll = unroll
        main(args)
    # for batch_size in [64, 128, 256, 512, 1024]:
    # for batch_size in [256, 512]:
    #     for num_timesteps in [50, 100, 150, 200, 250, 300]:
    #         for layer in [64, 128, 256, 512, 1024]:
    #             for layer_num in [1, 2, 3, 4, 5, 6]:
    #                 layers = [layer] * layer_num
    #                 n = 0
    #                 while n <= 10:
    #                     if n == 0:
    #                         unroll = 1
    #                     else:
    #                         unroll = math.ceil(0.1 * n * num_timesteps)
    #                     args = Args(batch_size=batch_size, 
    #                         num_timesteps=num_timesteps, 
    #                         num_iters=num_iters, 
    #                         layers=layers, 
    #                         unroll=unroll)
    #                     n += 1
    #                     main(args=args)


    # print arguments
    # for arg in vars(args):
    #     print(arg, ":", getattr(args, arg))
    
    # main(args=args)
