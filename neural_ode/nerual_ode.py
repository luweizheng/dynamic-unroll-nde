import time
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
from dataclasses import dataclass
import sys; 
sys.path.insert(0, '..')

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func
    hidden_size:int
    width_size:int
    depth:int

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = data_size
        self.width_size = width_size
        self.depth = depth
        self.func = Func(data_size, width_size, depth, key=key)
    
    
    def step(self, carry):
        (i, t0, dt, y0) = carry
        t = t0 + i * dt
        
        dy = dt * self.func(t, y0)
        y1 = y0 + dy
        carry = (i+1, t0, dt, y1)
        return (carry , y1)
        
    def __call__(self, ts, y0, unroll=1):
        t0=ts[0]
        dt0=ts[1] - ts[0]
        y0=y0
        carry = (0, t0, dt0, y0)
        
        def step_fn(carry, inp=None):
            del inp
            return self.step(carry)
        
        _, ys = jax.lax.scan(step_fn, carry, xs=None, length=len(ts), unroll=unroll)
        
        return ys
    


def _get_data(ts, *, key):
    y0 = jrandom.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        x = y / (1 + y)
        return jnp.stack([x[1], -x[0]], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


def get_data(dataset_size, num_timesteps, *, key):
    ts = jnp.linspace(0, 10, num_timesteps)
    key = jrandom.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


@eqx.filter_value_and_grad
def grad_loss(model, ti, yi, unroll):
    y_pred = jax.vmap(model, in_axes=(None, 0, None))(ti, yi[:, 0], unroll)
    return jnp.mean((yi - y_pred) ** 2)

@eqx.filter_jit
def make_step(ti, yi, model, optim, opt_state, unroll):
    loss, grads = grad_loss(model, ti, yi, unroll)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def train(args):
    key = jrandom.PRNGKey(args.seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)

    ts, ys = get_data(args.dataset_size, args.num_timesteps, key=data_key)
    _, length_size, data_size = ys.shape
    _ts = ts[: int(length_size * args.length)]
    _ys = ys[:, : int(length_size * args.length)]
    model = NeuralODE(data_size, args.width_size, args.depth, key=model_key)
    optim = optax.adabelief(args.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    
    for step, (yi,) in zip(
            range(args.num_iters), dataloader((_ys,), args.batch_size, key=loader_key)
        ):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, optim ,opt_state, args.unroll)
            end = time.time()
            if (step % args.print_every) == 0 or step == args.num_iters - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
    if args.plot:
        plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
        plt.plot(ts, ys[0, :, 1], c="dodgerblue")
        model_y = model(ts, ys[0, 0])
        plt.plot(ts, model_y[:, 0], c="crimson", label="Model")
        plt.plot(ts, model_y[:, 1], c="crimson")
        plt.legend()
        plt.tight_layout()
        plt.savefig("neural_ode.png")
        plt.show()

@dataclass
class Args:
    batch_size: int
    lr:float
    # dim of SDE
    dataset_size: int
    
    num_timesteps: int
    num_iters: int
    length:int
    
    # network
    depth: int
    width_size: int
    
    # dynamic unroll
    unroll: int 
    seed:int
    
    #plot
    plot:bool = True
    print_every: int = 100

def main():
    
    args = Args (
            batch_size=32,
            lr=3e-3,
            dataset_size=256,
            num_timesteps=1000,
            num_iters=500,
            length=1,
            depth=4,
            width_size=64,
            unroll=1,
            seed=5678)
    train(args)

if __name__ == '__main__':
    main()