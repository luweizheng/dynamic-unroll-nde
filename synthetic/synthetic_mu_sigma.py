import time
from typing import Union
from dataclasses import dataclass
from functools import partial

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

def lipswish(x):
    return 0.909 * jnn.silu(x)


class MuField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, scale, *, key, **kwargs):
        super().__init__(**kwargs)
        scale_key, mlp_key = jrandom.split(key)
        if scale:
            self.scale = jrandom.uniform(
                scale_key, (hidden_size,), minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t, y):
        return self.scale * self.mlp(jnp.concatenate([t, y]))


class SigmaField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(
        self, control_size, hidden_size, width_size, depth, scale, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        scale_key, mlp_key = jrandom.split(key)
        if scale:
            self.scale = jrandom.uniform(
                scale_key, (hidden_size, control_size), minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = control_size
        self.hidden_size = hidden_size

    def __call__(self, t, y):
        return self.scale * self.mlp(jnp.concatenate([t, y])).reshape(
            self.hidden_size, self.control_size
        )


class SDEStep(eqx.Module):
    mf: MuField  # drift
    sf: SigmaField  # diffusion

    def __init__(
        self,
        noise_size,
        hidden_size,
        width_size,
        depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mf_key, sf_key = jrandom.split(key, 2)

        self.mf = MuField(hidden_size, width_size, depth, scale=True, key=mf_key)
        self.sf = SigmaField(
            noise_size, hidden_size, width_size, depth, scale=True, key=sf_key
        )

        self.noise_size = noise_size

    def __call__(self, carry, input=None):
        (i, t0, dt, y0, key) = carry
        t = jnp.full((1, ), t0 + i * dt)
        _key1, _key2 = jrandom.split(key, 2)
        bm = jrandom.normal(_key1, (self.noise_size, )) * jnp.sqrt(dt)
        drift_term = self.vf(t=t, y=y0) * dt
        diffusion_term = jnp.dot(self.cvf(t=t, y=y0), bm)
        y1 = y0 + drift_term + diffusion_term
        carry = (i+1, t0, dt, y1, _key2)

        return carry, y1

class NeuralSDE(eqx.Module):
    initial: eqx.nn.MLP
    step: SDEStep
    readout: eqx.nn.Linear
    initial_noise_size: int
    noise_size: int


    def __init__(
        self,
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        initial_key, step_key, readout_key = jrandom.split(key, 3)

        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )

        self.step = SDEStep(noise_size=noise_size, hidden_size=hidden_size, width_size=width_size, depth=depth, key=step_key)
        self.readout = eqx.nn.Linear(hidden_size, data_size, key=readout_key)

        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size


    def __call__(self, ts, *, key):
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = 1.0
        init_key, bm_key = jrandom.split(key, 2)
        init = jrandom.normal(init_key, (self.initial_noise_size,))

        y0 = self.initial(init)

        ys = solve(self.step_fn, y0, t0, dt0, 64, bm_key)
        
        result = jax.vmap(self.readout)(ys)
        
        return result



def solve(step, y0, t0, dt, num_steps, bm_key):
    carry = (0, t0, dt, y0, bm_key)

    _, ys = jax.lax.scan(step, carry, xs=None, length=num_steps)

    return ys
        

@partial(jit)
def train_step(step, train_state, y0, dW, ts, times, unroll):

    def loss_fn(params):

        # @partial(jit)
        # def forward_fn(params, y0, dW, ts, times):
        #     ys = train_state.apply_fn({'params': params}, y0, dW, ts, times)
        #     return ys

        # ys = forward_fn(params, y0, dW, ts, times)
        ys = train_state.apply_fn({'params': params}, y0, dW, ts, times)
        # dummy loss
        loss = jnp.sum(jnp.mean(ys, axis=0))

        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss

def train(args):

    rng = jrandom.PRNGKey(42)

    dt = args.T / args.num_timesteps
    dummy_dW = jrandom.normal(rng, (args.num_timesteps, args.batch_size, args.dim))
    times = jnp.arange(0, args.T, dt)
    y0 = jnp.ones((args.batch_size, args.dim), jnp.float32)
    ts = jnp.linspace(0, args.T, args.num_ts)

    # make features
    output = ""
    cell = SDEStep(dim=args.dim, layers=args.layers, dt=dt, noise='diagonal')

    dummy_carry = (0, jnp.zeros((args.batch_size, args.dim)))
    cell_variable = cell.init(rng, carry=dummy_carry, dW=dummy_dW[0])

    hlo_module = jax.xla_computation(cell.apply)({'params': cell_variable['params']}, carry=dummy_carry, dW=dummy_dW[0]).as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    step_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
    step_bytes_access_gb = step_cost['bytes accessed'] / 1e9
    step_flops_g = step_cost['flops'] / 1e9
    
    # step bytes access in GB
    output = output + str(step_bytes_access_gb) + ','
    # step G FLOPS 
    output = output + str(step_flops_g) + ','
    # step Arithmetic Intensity
    output = output + str(step_flops_g / step_bytes_access_gb) + ','

    
    total_params = sum(p.size for p in jax.tree_leaves(cell_variable['params']))

    mdl_kwargs = {'dim': args.dim, 'dt': dt, 'noise': "diagonal", 'layers': args.layers}
    sde = ForwardSDE(SDEStep, mdl_kwargs, dt, unroll=args.unroll)

    variables = sde.init(rng, y0, dummy_dW, ts, times)

    hlo_module = jax.xla_computation(sde.apply)({'params': variables['params']}, y0, dummy_dW, ts, times).as_hlo_module()
    full_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
    full_bytes_access_gb = full_cost['bytes accessed'] / 1e9
    full_flops_g = full_cost['flops'] / 1e9
    # step bytes access in GB
    output = output + str(full_bytes_access_gb) + ','
    # step G FLOPS 
    output = output + str(full_flops_g) + ','
    # step Arithmetic Intensity
    output = output + str(full_flops_g / full_bytes_access_gb) + ','

    #  params size in million
    output = output + str(total_params / 1e6) + ','

    # input dimension
    output = output + str(args.dim) + ','
    # number of layers
    output = output + str(len(args.layers)) + ','

    # device 1 Tesla V100 2 Titan RTX 
    output = output + str(2) + ','
    # params share across different iteration, 0 false
    output = output + str(0) + ','

    # batch size
    output = output + str(args.batch_size) + ","
    # unroll factor
    unroll_factor = args.unroll / args.num_timesteps
    output = output + str(unroll_factor) + ","
    # number of timesteps
    output = output + str(args.num_timesteps) + ","

    # type of sde 0 foward sde 1 foward-backward sde
    output = output + str(args.num_timesteps) + ","

    learning_rate = 1e-2
    learning_rate_fn = optax.exponential_decay(learning_rate, 1, 0.999)
    tx = optax.adam(learning_rate=learning_rate_fn)

    state = TrainState.create(
        apply_fn=sde.apply,
        params=variables['params'],
        tx=tx
    )

    start_time = time.time()

    for step in range(args.num_iters):
        rng, _ = jax.random.split(rng)
        dW = jrandom.normal(rng, (args.num_timesteps, args.batch_size, args.dim))
        state, loss = train_step(step, state, y0, dW, ts, times, unroll=args.unroll)

        if step == 0:
            compile_time = time.time()
            iter_time = time.time()
        # if step % 100 == 0 and step > 0:
        #     iter_time_list.append(time.time() - iter_time)
        #     iter_time = time.time()


    output = output + str(compile_time - start_time) + ','
    output = output + str(time.time() - compile_time)
    print(output)


@dataclass
class Args:
    batch_size: int
    dim: int
    num_timesteps: int
    num_ts: int
    num_iters: int
    layers: Sequence[int]
    unroll: int
    T: float = 1.0


def main(args):
    train(args)


if __name__ == '__main__':
    # args = Args(batch_size=512, 
    #                         dim=2,
    #                         num_timesteps=1000,
    #                         num_ts=10,
    #                         num_iters=1000, 
    #                         layers=[256, 256, dim], 
    #                         unroll=1)
    # main(args=args)
    for batch_size in [256]:
    # for batch_size in [512]:
        for num_timesteps in [200, 250, 300]:
            for layer in [128, 256, 512, 1024]:
                for layer_num in [4, 5, 6]:
                    for dim in [4, 16, 32, 64]:
                        layers = [layer] * layer_num + [dim]
                        n = 0
                        while n <= 5:
                            if n == 0:
                                unroll = 1
                            else:
                                unroll = math.ceil(0.1 * n * num_timesteps)
                                if unroll > 100:
                                    break
                            args = Args(batch_size=batch_size, 
                                dim=dim,
                                num_timesteps=num_timesteps,
                                num_ts=10,
                                num_iters=1000, 
                                layers=layers, 
                                unroll=unroll)
                            n += 1
                            main(args=args)