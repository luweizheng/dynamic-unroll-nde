import time
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import argparse

matplotlib.rcParams.update({"font.size": 30})


class Func(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(y)


class LatentODE(eqx.Module):
    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int
    
    diffrax_solver: bool
    unroll: int

    def __init__(
        self, *, data_size, hidden_size, latent_size, width_size, depth, key, diffrax_solver=False, unroll=1, **kwargs
    ):
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jrandom.split(key, 5)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)

        self.hidden_to_latent = eqx.nn.Linear(
            hidden_size, 2 * latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)

        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.diffrax_solver = diffrax_solver
        self.unroll = unroll
    # Encoder of the VAE
    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size:]
        std = jnp.exp(logstd)
        latent = mean + jrandom.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    # Decoder of the VAE
    def _sample(self, ts, latent):
        dt0 = 0.4  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        
        t0 = ts[0]
        carry = (0, t0, dt0, y0)
        
        def step_fn(carry):
            (i, t0, dt, y0) = carry
            t = t0 + i * dt

            dy = dt * self.func(t, y0)
            y1 = y0 + dy
            carry = (i+1, t0, dt, y1)
            return (carry, y1)
        
        if self.diffrax_solver:
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(self.func),
                diffrax.Tsit5(),
                ts[0],
                ts[-1],
                dt0,
                y0,
                saveat=diffrax.SaveAt(ts=ts),
            )
            ys = sol.ys
        else:
            _, ys = jax.lax.scan(step_fn, carry, xs=None,
                                    length=len(ts), unroll=self.unroll)
        return jax.vmap(self.hidden_to_data)(ys)
        

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * \
            jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    # Run both encoder and decoder during training.
    def train(self, ts, ys, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        return self._loss(ys, pred_ys, mean, std)

    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        latent = jrandom.normal(key, (self.latent_size,))
        return self._sample(ts, latent)


def get_data(dataset_size, num_timesteps, *, key):
    ykey, tkey1, tkey2 = jrandom.split(key, 3)

    y0 = jrandom.normal(ykey, (dataset_size, 2))

    t0 = 0
    t1 = 2 + jrandom.uniform(tkey1, (dataset_size,))
    ts = jrandom.uniform(tkey2, (dataset_size, num_timesteps)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    def func(t, y, args):
        return jnp.array([[-0.1, 1.3], [-1, -0.1]]) @ y

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    ys = jax.vmap(solve)(ts, y0)

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
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

def train(args):
    key = jrandom.PRNGKey(args.seed)
    data_key, model_key, loader_key, train_key, sample_key = jrandom.split(key, 5)

    ts, ys = get_data(args.dataset_size, args.num_timesteps, key=data_key)

    model = LatentODE(
        data_size=ys.shape[-1],
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        width_size=args.width_size,
        depth=args.depth,
        key=model_key,
        diffrax_solver=args.diffrax_solver,
        unroll=args.unroll
    )

    @eqx.filter_value_and_grad
    def loss(model, ts_i, ys_i, key_i):
        batch_size, _ = ts_i.shape
        key_i = jrandom.split(key_i, batch_size)
        loss = jax.vmap(model.train)(ts_i, ys_i, key=key_i)
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(model, opt_state, ts_i, ys_i, key_i):
        value, grads = loss(model, ts_i, ys_i, key_i)
        key_i = jrandom.split(key_i, 1)[0]
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, key_i

    optim = optax.adam(args.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Plot results
    # if args.plot:
    #     num_plots = 1 + (args.num_iters - 1) // args.save_every
    #     if ((args.num_iters - 1) % args.save_every) != 0:
    #         num_plots += 1
    #     fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 8, 8))
    #     axs[0].set_ylabel("x")
    #     axs = iter(axs)
    
    start_ts = time.time()
    for step, (ts_i, ys_i) in zip(
        range(args.num_iters), dataloader((ts, ys), args.batch_size, key=loader_key)
    ):
        cal_start = time.time()
        value, model, opt_state, train_key = make_step(
            model, opt_state, ts_i, ys_i, train_key
        )
        if step == 0:
            compile_ts = time.time()
        if (step % args.print_every) == 0 or step == args.num_iters - 1:
            cal_end = time.time()
            print(
                f"Step: {step}, Loss: {value}, Computation time: {cal_end - cal_start}")
            
        # if args.plot:
        #     if (step % args.save_every) == 0 or step == args.num_iters - 1:
        #         ax = next(axs)
        #         # Sample over a longer time interval than we trained on. The model will be
        #         # sufficiently good that it will correctly extrapolate!
        #         sample_t = jnp.linspace(0, 12, 300)
        #         sample_y = model.sample(sample_t, key=sample_key)
        #         sample_t = np.asarray(sample_t)
        #         sample_y = np.asarray(sample_y)
        #         ax.plot(sample_t, sample_y[:, 0])
        #         ax.plot(sample_t, sample_y[:, 1])
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         ax.set_xlabel("t")
    # if args.plot:
    #     plt.savefig("latent_ode.png")
    #     plt.show()
    
    if args.print_time_use:
        compile_time = compile_ts - start_ts
        run_time = time.time() - compile_ts
        print(f"unroll: {args.unroll}, compiel_time: {compile_time}, run_time: {run_time * 50}, total_time: {compile_time + run_time * 50}")
            
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--dataset-size', type=int, default=10000)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--width-size', type=int, default=16)
    parser.add_argument('--latent-size', type=int, default=16)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--num-timesteps', type=int, default=100)
    parser.add_argument('--num-iters', type=int, default=500)
    parser.add_argument('--unroll', type=int, default=1)
    parser.add_argument('--seed', type=int, default=5678)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--print-every', type=int, default=200)
    parser.add_argument('--diffrax-solver', action='store_true')
    parser.add_argument('--print-time-use', action='store_true')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
