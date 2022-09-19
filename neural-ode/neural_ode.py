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
import jax.tree_util as jtu
import argparse


class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args=None):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func
    hidden_size: int
    width_size: int
    depth: int
    unroll: int
    diffrax_solver: bool

    def __init__(self, data_size, width_size, depth, key, diffrax_solver=False, unroll=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = data_size
        self.width_size = width_size
        self.depth = depth
        self.func = Func(data_size, width_size, depth, key=key)
        self.diffrax_solver = diffrax_solver
        self.unroll = unroll

    def step(self, carry):
        (i, t0, dt, y0) = carry
        t = t0 + i * dt

        dy = dt * self.func(t, y0)
        y1 = y0 + dy
        carry = (i+1, t0, dt, y1)
        return (carry, y1)

    def __call__(self, ts, y0):
        t0 = ts[0]
        dt0 = ts[1] - ts[0]
        y0 = y0
        carry = (0, t0, dt0, y0)

        def step_fn(carry, inp=None):
            del inp
            return self.step(carry)
        

        if self.diffrax_solver:
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(self.func),
                diffrax.Euler(),
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0],
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
            )
            ys = solution.ys
        else:
            _, ys = jax.lax.scan(step_fn, carry, xs=None,
                                    length=len(ts), unroll=self.unroll)

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
def grad_loss(model, ti, yi):
    y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
    return jnp.mean((yi - y_pred) ** 2)


@eqx.filter_jit
def make_step(ti, yi, model, optim, opt_state):
    loss, grads = grad_loss(model, ti, yi)
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
    print(args)
    model = NeuralODE(data_size, args.width_size, args.depth,
                      key=model_key, diffrax_solver=args.diffrax_solver, unroll=args.unroll)
    optim = optax.adabelief(args.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    start_ts = time.time()
    for step, (yi,) in zip(
        range(args.num_iters), dataloader(
            (_ys,), args.batch_size, key=loader_key)
    ):
        loss, model, opt_state = make_step(
            _ts, yi, model, optim, opt_state)
        if step == 0:
            compile_ts = time.time()
        if (step % args.print_every) == 0 or step == args.num_iters - 1:
            end_ts = time.time()
            print(
                f"Step: {step}, Loss: {loss}, Computation time: {end_ts - start_ts}")

    if args.print_time_use:
        compile_time = compile_ts - start_ts
        run_time = time.time() - compile_ts
        print(f"unroll: {args.unroll}, compiel_time: {compile_time}, run_time: {run_time * 50}, total_time: {compile_time + run_time * 50}")

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    # hidden_size == dataset_size
    parser.add_argument('--dataset-size', type=int, default=256)
    parser.add_argument('--width-size', type=int, default=64)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--length', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=200)
    parser.add_argument('--num-iters', type=int, default=1000)
    parser.add_argument('--unroll', type=int, default=1)
    parser.add_argument('--seed', type=int, default=5678)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--print-every', type=int, default=200)
    parser.add_argument('--diffrax-solver', action='store_true')
    parser.add_argument('--print-time-use', action='store_true')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
