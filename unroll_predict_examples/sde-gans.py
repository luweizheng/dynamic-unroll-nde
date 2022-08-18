import time
from typing import Union, Callable
import xgboost as xgb
import numpy as np
import diffrax
from functools import partial
from diffrax.misc import ω
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import argparse
from dataclasses import dataclass
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import sys; 
sys.path.insert(0, '..')
from simulated_annealing import annealing

@dataclass
class Args:
    # network
    initial_noise_size:int
    noise_size:int
    hidden_size:int
    width_size:int
    depth: int
    
    generator_lr:float
    discriminator_lr:float
    
    batch_size: int
    steps:int
    steps_per_print:int

    dataset_size:int
    seed:int
    
    # dynamic unroll
    unroll1: int
    unroll2: int

def lipswish(x):
    return 0.909 * jnn.silu(x)

class VectorField(eqx.Module):
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

    def __call__(self, t, y, args):
        return self.scale * self.mlp(jnp.concatenate([t[None], y]))


class ControlledVectorField(eqx.Module):
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

    def __call__(self, t, y, args):
        return self.scale * self.mlp(jnp.concatenate([t[None], y])).reshape(
            self.hidden_size, self.control_size
        )


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
    noise_size: int
    mf: MuField  # drift
    sf: SigmaField  # diffusion
    bm: diffrax.VirtualBrownianTree

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
        mf_key, sf_key, bm_key = jrandom.split(key, 3)

        self.mf = MuField(hidden_size, width_size, depth, scale=True, key=mf_key)
        self.sf = SigmaField(
            noise_size, hidden_size, width_size, depth, scale=True, key=sf_key
        )
        self.noise_size = noise_size
        self.bm = diffrax.VirtualBrownianTree(
            t0=0.0, t1=63.0, tol=1.0 / 2, shape=(noise_size,), key=bm_key
        )

    def __call__(self, carry, input=None):
        (i, t0, dt, y0, key) = carry
        t = jnp.full((1, ), t0 + i * dt)
        _key1, _key2 = jrandom.split(key, 2)
        drift_term = self.mf(t=t, y=y0) * dt
        bm = jrandom.normal(_key1, (self.noise_size, )) * jnp.sqrt(dt)
        diffusion_term = jnp.dot(self.sf(t=t, y=y0), bm)
        y1 = y0 + drift_term + diffusion_term
        carry = (i+1, t0, dt, y1, _key2)

        return carry, y1
    

class CDEStep(eqx.Module):
    
    
    terms: diffrax.MultiTerm
    
    
    def __init__(self, terms) -> None:
        super().__init__()
        self.terms = terms
        
    
    def __call__(self, carry, input=None):
        (i, t0, dt, y0, yhat0 ,vf0) = carry

        t1 = t0 + dt
        
        # extract key steps from  diffrax.ReversibleHeun
        vf0 = jax.lax.cond(False, lambda _: self.terms.vf(t0, y0, args=None), lambda _: vf0, None)
        control = self.terms.contr(t0, t1)
        yhat1 = (2 * y0**ω - yhat0**ω + self.terms.prod(vf0, control) ** ω).ω
        vf1 = self.terms.vf(t1, yhat1, args=None)
        y1 = (y0**ω + 0.5 * self.terms.prod((vf0**ω + vf1**ω).ω, control) ** ω).ω
        carry = (i + 1, t1, dt, y1, yhat1, vf1)
        return carry, y1

class NeuralSDE(eqx.Module):
    initial: eqx.nn.MLP
    step: SDEStep
    readout: eqx.nn.Linear
    initial_noise_size: int
    noise_size: int
    hidden_size:int
    depth:int

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
        self.hidden_size = hidden_size
        self.depth = depth
        
    
    def make_cost_model_feature(self):
        def step_fn(carry, inp):
            return self.step(carry, inp)
        
        dummy_t0 = 0.0
        dummy_dt = 0.2

        dummy_init_key, dummy_bm_key = jrandom.split(jrandom.PRNGKey(0), 2)
        
        dummy_init = jrandom.normal(dummy_init_key, (self.initial_noise_size,))
        y0 = self.initial(dummy_init)
        carry = (0, dummy_t0, dummy_dt, y0, dummy_bm_key)

        hlo_module = jax.xla_computation(step_fn)(carry, None).as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        step_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
        step_bytes_access = step_cost['bytes accessed']
        step_bytes_access_op0 = step_cost['bytes accessed operand 0 {}']
        step_bytes_access_op1 = step_cost['bytes accessed operand 1 {}']
        step_bytes_access_out = step_cost['bytes accessed output {}']
        step_flops = step_cost['flops']
        
        features = []
        features.append(step_bytes_access)
        features.append(step_bytes_access_op0)
        features.append(step_bytes_access_op1)
        features.append(step_bytes_access_out)
        features.append(step_flops)
        features.append(step_flops / step_bytes_access)
        total_params = sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(self.step, eqx.is_array)))
        features.append(total_params / 1e6)
        features.append(self.hidden_size)
        features.append(self.depth )
        features.append(self.depth )
        features.append(0)
        features.append(0)
        features.append(0)

        return features

    def __call__(self, ts, key, unroll):
        t0 = ts[0]
        dt0 = 0.1
        init_key, bm_key = jrandom.split(key, 2)
        init = jrandom.normal(init_key, (self.initial_noise_size,))

        y0 = self.initial(init)

        def step_fn(carry, input):
            return self.step(carry, input)

        
        ys = solve(step_fn, y0, t0, dt0, len(ts), bm_key, unroll=unroll)
        
        result = jax.vmap(self.readout)(ys)
        
        return result


def solve(step, y0, t0, dt, num_steps, bm_key, unroll=1):
    carry = (0, t0, dt, y0, bm_key)

    _, ys = jax.lax.scan(step, carry, xs=None, length=num_steps, unroll=unroll)

    return ys

def solve_cde(step, y0, t0, dt, vf0, num_steps, unroll=1):
    carry = (0, t0, dt, y0, y0, vf0)
    
    _, ys = jax.lax.scan(step, carry, xs=None, length=num_steps, unroll=unroll)
    
    return ys



class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    data_size:int
    hidden_size:int
    width_size:int
    depth:int
    vf: VectorField
    cvf: ControlledVectorField
    readout: eqx.nn.Linear
    

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jrandom.split(key, 4)

        self.initial = eqx.nn.MLP(
            data_size + 1, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=False, key=vf_key)
        self.cvf = ControlledVectorField(
            data_size, hidden_size, width_size, depth, scale=False, key=cvf_key
        )
        
        self.readout = eqx.nn.Linear(hidden_size, 1, key=readout_key)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.width_size = width_size
        self.depth = depth

    def make_cost_model_feature(self, num_timesteps):
        
        dummy_ts = jnp.linspace(0, 1, num_timesteps)
        dummy_ys = jnp.ones((dummy_ts.shape[0], 1))
        dummy_ys = diffrax.linear_interpolation(
            dummy_ts, dummy_ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([dummy_ts[0, None], dummy_ys[0]])
        control = diffrax.LinearInterpolation(dummy_ts, dummy_ys)
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        step = CDEStep(terms=terms)
        t0 = dummy_ts[0]
        dt0 = 0.1
        y0 = self.initial(init)
        vf0 = terms.vf(t0, y0, args=None)
        def step_fn(carry, inp):
            return step(carry, inp)

        
        carry = (0, t0, dt0, y0, y0, vf0)

        hlo_module = jax.xla_computation(step_fn)(carry, None).as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        step_cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo_module)
        step_bytes_access = step_cost['bytes accessed']
        step_bytes_access_op0 = step_cost['bytes accessed operand 0 {}']
        step_bytes_access_op1 = step_cost['bytes accessed operand 1 {}']
        step_bytes_access_out = step_cost['bytes accessed output {}']
        step_flops = step_cost['flops']
        
        features = []
        features.append(step_bytes_access)
        features.append(step_bytes_access_op0)
        features.append(step_bytes_access_op1)
        features.append(step_bytes_access_out)
        features.append(step_flops)
        features.append(step_flops / step_bytes_access)
        # discriminator has nothing to learn 
        total_params = 0
        features.append(total_params / 1e6)
        features.append(self.hidden_size)
        features.append(self.depth )
        features.append(self.depth )
        features.append(0)
        features.append(0)
        features.append(0)
        
        return features

    def __call__(self, ts, ys, unroll):
        # Interpolate data into a continuous path.
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([ts[0, None], ys[0]])
        control = diffrax.LinearInterpolation(ts, ys)
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        step = CDEStep(terms=terms)
        t0 = ts[0]
        dt0 = 0.1
        y0 = self.initial(init)
        
        def step_fn(carry, input):
            return step(carry=carry, input=input)
        
        vf0 = terms.vf(t0, y0, args=None)
        
        ys = solve_cde(step_fn, y0, t0, dt0, vf0, len(ts), unroll=unroll)
        
        result = jax.vmap(self.readout)(ys)
        
        return result

    @eqx.filter_jit
    def clip_weights(self):
        leaves, treedef = jax.tree_util.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
        )
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, eqx.nn.Linear):
                lim = 1 / leaf.out_features
                leaf = eqx.tree_at(
                    lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
                )
            new_leaves.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)


@partial(jax.jit, static_argnames=['num_timesteps'])
def get_data(key, num_timesteps):
    bm_key, y0_key, drop_key = jrandom.split(key, 3)

    mu = 0.02
    theta = 0.1
    sigma = 0.4

    t0 = 0
    t1 = 64.0
    t_size = num_timesteps

    def drift(t, y, args):
        return mu * t - theta * y

    def diffusion(t, y, args):
        return 2 * sigma * t / t1

    bm = diffrax.UnsafeBrownianPath(shape=(), key=bm_key)
    drift = diffrax.ODETerm(drift)
    diffusion = diffrax.ControlTerm(diffusion, bm)
    terms = diffrax.MultiTerm(drift, diffusion)
    solver = diffrax.Euler()
    dt0 = 0.1
    y0 = jrandom.uniform(y0_key, (1,), minval=-1, maxval=1)
    ts = jnp.linspace(t0, t1, num_timesteps)
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        terms, solver, t0, t1, dt0, y0, saveat=saveat, adjoint=diffrax.NoAdjoint()
    )

    # Make the data irregularly sampled
    to_drop = jrandom.bernoulli(drop_key, 0.3, (t_size, 1))
    ys = jnp.where(to_drop, jnp.nan, sol.ys)

    return ts, ys


def dataloader(arrays, batch_size, loop, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        key = jrandom.split(key, 1)[0]
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
        if not loop:
            break


@eqx.filter_jit
def loss(generator, discriminator, ts_i, ys_i, key, unroll1, unroll2, step=0):
    batch_size, _ = ts_i.shape
    key = jrandom.fold_in(key, step)
    key = jrandom.split(key, batch_size)
    fake_ys_i = jax.vmap(generator, in_axes=(0, 0, None))(ts_i, key, unroll1)
    real_score = jax.vmap(discriminator, in_axes=(0, 0, None))(ts_i, ys_i, unroll2)
    fake_score = jax.vmap(discriminator, in_axes=(0, 0, None))(ts_i, fake_ys_i, unroll2)
    return jnp.mean(real_score - fake_score)


@eqx.filter_grad
def grad_loss(g_d, ts_i, ys_i, key, unroll1, unroll2, step):
    generator, discriminator = g_d
    return loss(generator, discriminator, ts_i, ys_i, key, unroll1, unroll2, step)


def increase_update_initial(updates):
    get_initial_leaves = lambda u: jax.tree_util.tree_leaves(u.initial)
    return eqx.tree_at(get_initial_leaves, updates, replace_fn=lambda x: x * 10)


@eqx.filter_jit
def make_step(
    generator,
    discriminator,
    g_opt_state,
    d_opt_state,
    g_optim,
    d_optim,
    ts_i,
    ys_i,
    key,
    unroll1,
    unroll2,
    step,
):
    g_grad, d_grad = grad_loss((generator, discriminator), ts_i, ys_i, key, unroll1, unroll2, step)
    g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
    d_updates, d_opt_state = d_optim.update(d_grad, d_opt_state)
    g_updates = increase_update_initial(g_updates)
    d_updates = increase_update_initial(d_updates)
    generator = eqx.apply_updates(generator, g_updates)
    discriminator = eqx.apply_updates(discriminator, d_updates)
    discriminator = discriminator.clip_weights()
    return generator, discriminator, g_opt_state, d_opt_state


def train(
    xgb_dir='',
    initial_noise_size=5,
    noise_size=3,
    hidden_size=16,
    width_size=16,
    depth=1,
    generator_lr=2e-5,
    discriminator_lr=1e-4,
    batch_size=1024,
    num_timesteps=640,
    steps=1000,
    steps_per_print=200,
    dataset_size=8192,
    seed=5678,
    unroll1=1,
    unroll2=1,
    search_method="exhaustive",
):
    start_ts = time.time()
    key = jrandom.PRNGKey(seed)
    (
        data_key,
        generator_key,
        discriminator_key,
        dataloader_key,
        train_key,
        evaluate_key,
        sample_key,
    ) = jrandom.split(key, 7)
    data_key = jrandom.split(data_key, dataset_size)

    ts, ys = jax.vmap(get_data, in_axes=(0, None))(data_key, num_timesteps)
    _, _, data_size = ys.shape

    generator = NeuralSDE(
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        key=generator_key,
    )
    discriminator = NeuralCDE(
        data_size, hidden_size, width_size, depth, key=discriminator_key
    )
    
    gen_features = generator.make_cost_model_feature()
    gen_features.append(batch_size)
    gen_features.append(num_timesteps)
    dis_features = discriminator.make_cost_model_feature(num_timesteps)
    dis_features.append(batch_size)
    dis_features.append(num_timesteps)
    compile_model_loaded = xgb.Booster()

    compile_model_loaded.load_model(xgb_dir+"compile2.txt")

    run_model_loaded = xgb.Booster()
    run_model_loaded.load_model(xgb_dir+"run2.txt")

    def raw_cost_fn(features, unroll):
        cur_features = features + [unroll]
        
        compilation_time_pred = compile_model_loaded.predict(xgb.DMatrix([cur_features]))
        run_time_pred = run_model_loaded.predict(xgb.DMatrix([cur_features]))
        total_time_pred = compilation_time_pred + run_time_pred * 10
        
        return total_time_pred

    if search_method == "exhaustive":
        # exhaustively iterate a list of candidates
        unroll_list = [2, 5, 8, 10, 15, 20, 30, 40, 50]
        total_time_pred_list1 = []
        total_time_pred_list2 = []
        cost_fn1 = lambda gen_features, unroll: raw_cost_fn(gen_features, unroll)
        cost_fn2 = lambda dis_features, unroll: raw_cost_fn(dis_features, unroll)
        for unroll in unroll_list:
            total_time_pred1 = cost_fn1(gen_features, unroll)
            total_time_pred_list1.append(total_time_pred1)
            total_time_pred2 = cost_fn2(dis_features, unroll)
            total_time_pred_list2.append(total_time_pred2)
        predicted_unroll1 = unroll_list[np.argmin(total_time_pred_list1)]
        predicted_unroll2 = unroll_list[np.argmin(total_time_pred_list2)]
    elif search_method == "sa_scipy":
        # dual annealing from scipy
        bounds = [[2, num_timesteps // 2]]
        from scipy.optimize import dual_annealing

        cost_fn = lambda unroll: raw_cost_fn(gen_features, unroll)
        result = dual_annealing(cost_fn, bounds, maxiter=20)
        predicted_unroll = result['x']
    elif search_method == "sa_our":
        # 自己实现的SA算法还不太成熟
        # my own implementation of SA
        bounds = (2, num_timesteps // 2)

        def clip(x, bounds):
            """ Force x to be in the interval."""
            a, b = bounds
            return int(max(min(x, b), a))

        def random_neighbour(x, bounds, fraction=1):
            """Move a little bit x, from the left or the right."""
            amplitude = (max(bounds) - min(bounds)) * fraction / 10
            delta = (-amplitude/2.) + amplitude * np.random.random_sample()
            return clip(x + delta, bounds)
        predicted_unroll, _, _, _ = annealing(bounds, cost_fn, random_neighbour=random_neighbour, maxsteps=20, debug=False)
    
    print(f"predicted unroll_1: {predicted_unroll1}, predicted unroll_2: {predicted_unroll2}")

    # train
    # 为了测试上面的cost model，先把下面的注释，下面这部分太浪费时间
    g_optim = optax.rmsprop(generator_lr)
    d_optim = optax.rmsprop(-discriminator_lr)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_inexact_array))
    d_opt_state = d_optim.init(eqx.filter(discriminator, eqx.is_inexact_array))

    infinite_dataloader = dataloader(
        (ts, ys), batch_size, loop=True, key=dataloader_key
    )

    # start_time = time.time()
    # last_iter_time = start_time
    for step, (ts_i, ys_i) in zip(range(steps), infinite_dataloader):
        step = jnp.asarray(step)
        generator, discriminator, g_opt_state, d_opt_state = make_step(
            generator,
            discriminator,
            g_opt_state,
            d_opt_state,
            g_optim,
            d_optim,
            ts_i,
            ys_i,
            key,
            unroll1,
            unroll2,
            step,
        )
        if step == 0:
            compile_ts = time.time()
        if (step % steps_per_print) == 0 or step == steps - 1:
            total_score = 0
            num_batches = 0
            for ts_i, ys_i in dataloader(
                (ts, ys), batch_size, loop=False, key=evaluate_key
            ):
                score = loss(generator, discriminator, ts_i, ys_i, sample_key)
                total_score += score.item()
                num_batches += 1
            print(f"Step: {step}, Iter Time: {time.time() - last_iter_time: .3f}, Loss: {total_score / num_batches}")
            last_iter_time = time.time()

    # Plot samples
    fig, ax = plt.subplots()
    num_samples = min(50, dataset_size)
    ts_to_plot = ts[:num_samples]
    ys_to_plot = ys[:num_samples]

    def _interp(ti, yi):
        return diffrax.linear_interpolation(
            ti, yi, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )

    ys_to_plot = jax.vmap(_interp)(ts_to_plot, ys_to_plot)[..., 0]
    ys_sampled = jax.vmap(generator)(
        ts_to_plot, key=jrandom.split(sample_key, num_samples)
    )[..., 0]
    kwargs = dict(label="Real")
    for ti, yi in zip(ts_to_plot, ys_to_plot):
        ax.plot(ti, yi, c="dodgerblue", linewidth=0.5, alpha=0.7, **kwargs)
        kwargs = {}
    kwargs = dict(label="Generated")
    for ti, yi in zip(ts_to_plot, ys_sampled):
        ax.plot(ti, yi, c="crimson", linewidth=0.5, alpha=0.7, **kwargs)
        kwargs = {}
    ax.set_title(f"{num_samples} samples from both real and generated distributions.")
    fig.legend()
    fig.tight_layout()
    fig.savefig("./my_control_5000_neural_sde.png")
    plt.show()
    compile_time = compile_ts - start_ts
    run_time = time.time() - compile_ts
    total_time = compile_time + run_time * 10

    print(f"unroll: {unroll}, actuall time: {total_time}")
    
    del generator
    del discriminator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xgb-dir', type=str, default='../cost-model/ckpt/')
    parser.add_argument('--initial-noise-size', type=int, default=5)
    parser.add_argument('--noise-size', type=int, default=3)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--width-size', type=int, default=500)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--generator-lr', type=float, default=2e-5)
    parser.add_argument('--discriminator-lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-timesteps', type=int, default=1000) 
    parser.add_argument('--steps', type=int, default=100, help="num_iters")
    parser.add_argument('--steps-per-print', type=int, default=200)
    parser.add_argument('--dataset-size', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=5678)
    parser.add_argument('--unroll1', type=int, default=1)
    parser.add_argument('--unroll2', type=int, default=1)
    parser.add_argument('--t0', type=float, default=0.0, required=False)
    parser.add_argument('--t1', type=float, default=64.0, required=False)
    unroll_list = [2, 5, 8, 10, 15, 20, 30, 40, 50]
    # test code
    args = parser.parse_args()
    # warm up run
    train(xgb_dir=args.xgb_dir,
          initial_noise_size=args.initial_noise_size,
          noise_size=args.noise_size,
          hidden_size=args.hidden_size,
          width_size=args.width_size,
          depth=args.depth,
          generator_lr=args.generator_lr,
          discriminator_lr=args.discriminator_lr,
          batch_size=args.batch_size,
          num_timesteps=args.num_timesteps,
          steps=args.steps,
          steps_per_print=args.steps_per_print,
          dataset_size=args.dataset_size,
          seed=args.seed,
          unroll1=args.unroll1,
          unroll2=args.unroll2)
    for unroll1 in unroll_list:
        args.unroll1 = unroll1
        for unroll2 in unroll_list:
            args.unroll2 = unroll2
            train(xgb_dir=args.xgb_dir,
                initial_noise_size=args.initial_noise_size,
                noise_size=args.noise_size,
                hidden_size=args.hidden_size,
                width_size=args.width_size,
                depth=args.depth,
                generator_lr=args.generator_lr,
                discriminator_lr=args.discriminator_lr,
                batch_size=args.batch_size,
                num_timesteps=args.num_timesteps,
                steps=args.steps,
                steps_per_print=args.steps_per_print,
                dataset_size=args.dataset_size,
                seed=args.seed,
                unroll1=args.unroll1,
                unroll2=args.unroll2)


if __name__ == '__main__':
    main()
