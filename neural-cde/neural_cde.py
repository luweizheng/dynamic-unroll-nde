import math
import time
from diffrax.misc import ω
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import matplotlib
import matplotlib.pyplot as plt
import argparse
import jax.tree_util as jtu
import optax  # https://github.com/deepmind/optax

matplotlib.rcParams.update({"font.size": 30})

class Func(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)

    
class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear
    diffrax_solver: bool
    unroll: int
    def __init__(self, data_size, hidden_size, width_size, depth, key, unroll, diffrax_solver=False ,**kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jrandom.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)
        self.unroll = unroll
        self.diffrax_solver = diffrax_solver
        
    def step(self, carry, term):
        (i, t0, dt, y0) = carry
        control = dt
        y1 = (y0**ω + term.vf_prod(t0, y0, args=None, control=control) ** ω).ω
        t1 = t0 + dt
        carry = (i+1, t1, dt, y1)
        return (carry , y1)

    def __call__(self, ts, coeffs, evolving_out=False, unroll=1):
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        dt0 = ts[1] - ts[0]
        y0 = self.initial(control.evaluate(ts[0]))
        carry = (0, ts[0], dt0, y0)
        
        def step_fn(carry, inp=None):
            del inp
            return self.step(carry, term)
        
        if self.diffrax_solver:
            solver = diffrax.Euler()
            if evolving_out:
                saveat = diffrax.SaveAt(ts=ts)
            else:
                saveat = diffrax.SaveAt(t1=True)
            solution = diffrax.diffeqsolve(
                term,
                solver,
                ts[0],
                ts[-1],
                dt0,
                y0,
                saveat=saveat,
            )
            ys = solution.ys
        else:
            _, ys = jax.lax.scan(step_fn, carry, xs=None, length=len(ts), unroll=self.unroll)
        
        if evolving_out:
            prediction = jax.vmap(lambda y: jnn.sigmoid(self.linear(y))[0])(ys)
        else:
            (prediction,) = jnn.sigmoid(self.linear(ys[-1]))
        return prediction

def get_data(dataset_size, add_noise, *, key):
    theta_key, noise_key = jrandom.split(key, 2)
    length = 100
    theta = jrandom.uniform(theta_key, (dataset_size,), minval=0, maxval=2 * math.pi)
    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    ts = jnp.broadcast_to(jnp.linspace(0, 4 * math.pi, length), (dataset_size, length))
    matrix = jnp.array([[-0.3, 2], [-2, -0.3]])
    ys = jax.vmap(
        lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
    )(y0, ts)
    ys = jnp.concatenate([ts[:, :, None], ys], axis=-1)  # time is a channel
    ys = ys.at[: dataset_size // 2, :, 1].multiply(-1)
    if add_noise:
        ys = ys + jrandom.normal(noise_key, ys.shape) * 0.1
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    labels = jnp.zeros((dataset_size,))
    labels = labels.at[: dataset_size // 2].set(1.0)
    _, _, data_size = ys.shape
    return ts, coeffs, labels, data_size

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

def train(args):
    key = jrandom.PRNGKey(args.seed)
    train_data_key, test_data_key, model_key, loader_key = jrandom.split(key, 4)

    ts, coeffs, labels, data_size = get_data(
        args.dataset_size, args.add_noise, key=train_data_key
    )

    model = NeuralCDE(data_size, args.hidden_size, args.width_size, args.depth, key=model_key, unroll=args.unroll, diffrax_solver=args.diffrax_solver)


    @eqx.filter_jit
    def loss(model, ti, label_i, coeff_i):
        pred = jax.vmap(model)(ti, coeff_i)
        # Binary cross-entropy
        bxe = label_i * jnp.log(pred) + (1 - label_i) * jnp.log(1 - pred)
        bxe = -jnp.mean(bxe)
        acc = jnp.mean((pred > 0.5) == (label_i == 1))
        return bxe, acc

    grad_loss = eqx.filter_value_and_grad(loss, has_aux=True)

    @eqx.filter_jit
    def make_step(model, data_i, opt_state):
        ti, label_i, *coeff_i = data_i
        (bxe, acc), grads = grad_loss(model, ti, label_i, coeff_i)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return bxe, acc, model, opt_state

    optim = optax.adam(args.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    start_ts = time.time()
    for step, data_i in zip(
        range(args.num_iters), dataloader((ts, labels) + coeffs, args.batch_size, key=loader_key)
    ):
        start = time.time()
        bxe, acc, model, opt_state = make_step(model, data_i, opt_state)
        if step == 0:
            compile_ts = time.time()
        # if (step % args.print_every) == 0 or step == args.num_iters - 1:
        #     end = time.time()
        #     print(
        #         f"Step: {step}, Loss: {bxe}, Accuracy: {acc}, Computation time: "
        #         f"{end - start}"
        #     )
    if args.print_time_use:
        end_ts = time.time()
        compile_time = compile_ts - start_ts
        run_time = end_ts - compile_ts
        print(f"unroll: {args.unroll}, compile_time: {compile_time},run_time: {run_time * 50}, total_time: {compile_time + run_time * 50}")

    # ts, coeffs, labels, _ = get_data(args.dataset_size, args.add_noise, key=test_data_key)
    # bxe, acc = loss(model, ts, labels, coeffs)
    # print(f"Test loss: {bxe}, Test Accuracy: {acc}")

    # Plot results
    if args.plot:
        sample_ts = ts[-1]
        sample_coeffs = tuple(c[-1] for c in coeffs)
        pred = model(sample_ts, sample_coeffs, evolving_out=True)
        interp = diffrax.CubicInterpolation(sample_ts, sample_coeffs)
        values = jax.vmap(interp.evaluate)(sample_ts)
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax1.plot(sample_ts, values[:, 1], c="dodgerblue")
        ax1.plot(sample_ts, values[:, 2], c="dodgerblue", label="Data")
        ax1.plot(sample_ts, pred, c="crimson", label="Classification")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel("t")
        ax1.legend()
        ax2.plot(values[:, 1], values[:, 2], c="dodgerblue", label="Data")
        ax2.plot(values[:, 1], values[:, 2], pred, c="crimson", label="Classification")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("Classification")
        plt.tight_layout()
        plt.savefig("neural_cde2.png")
        plt.show()
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--dataset-size', type=int, default=256)
    parser.add_argument('--add-noise', type=bool, default=False)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--width-size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
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