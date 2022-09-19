import math
import time
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax 
import argparse
from typing import Sequence, Callable
import equinox as eqx
import diffrax
import sys; 
sys.path.insert(0, '..')
from simulated_annealing import annealing

class FNN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size, out_size, width_size=width_size, depth=depth, key=key
        )

    def __call__(self, t, x, train: bool = True):
        y, f_vjp = jax.vjp(lambda x: self.mlp(jnp.concatenate([t, x])), x)
        dudx = f_vjp(jnp.ones(y.shape))
        return y, dudx[0]


class FBSDEStep(eqx.Module):
    unet: FNN
    noise_size: int
    mu_fn: Callable = eqx.static_field()
    sigma_fn: Callable = eqx.static_field()
    phi_fn: Callable = eqx.static_field()

    def __init__(self, in_size, out_size, width_size, depth, noise_size, key):
        self.unet = FNN(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, key=key)
        def mu_fn(t, X, Y, Z):
            del t, Y, Z
            return jnp.zeros_like(X)

        def sigma_fn(t, X, Y):
            del t, Y
            return 0.4 * X

        def phi_fn(t, X, Y, Z):
            del t
            return 0.05 * (Y - jnp.sum(X * Z, keepdims=True))
        
        self.mu_fn = mu_fn
        self.sigma_fn = sigma_fn
        self.phi_fn = phi_fn
        self.noise_size = noise_size
        

    def u_and_dudx(self, t, x):
        return self.unet(t, x)

    def __call__(self, carry, inp):
        # `t` and `W` are (batch_size, num_timestep, dim)
        # it have input data across iterations 

        (i, t0, dt, x0, y0, z0, key) = carry

        key, key2 = jrandom.split(key)

        # use `i` to index input data
        curr_t = jnp.full((1, ), t0 + i * dt)
        next_t = jnp.full((1, ), t0 + (i + 1) * dt)

        dW = jrandom.normal(key, (self.noise_size, )) * jnp.sqrt(dt)

        x1 = x0 + self.mu_fn(curr_t, x0, y0, z0) * dt + \
                self.sigma_fn(curr_t, x0, y0) * dW
    
        
        y1_tilde = y0 + self.phi_fn(curr_t, x0, y0, z0) * dt + \
            jnp.sum(z0 * self.sigma_fn(curr_t, x0, y0) * dW, keepdims=True)
        
        y1, z1 = self.unet(next_t, x1)

        carry = (i+1, t0, dt, x1, y1, z1, key2)
        outputs = (x1, y1_tilde, y1)
        return carry, outputs


class NeuralFBSDE(eqx.Module):
    step: FBSDEStep
    hidden_size: int
    depth: int
    width_size: int
    
    def __init__(self, in_size, out_size, width_size, depth, noise_size, key, diffrax_solver):
        self.step = FBSDEStep(in_size, out_size, width_size, depth, noise_size, key)
        self.hidden_size  = in_size - 1
        self.depth = depth
        self.width_size = width_size
        self.diffrax_solver = diffrax_solver


    def __call__(self, x0, t0, dt, num_timesteps, unroll=1, key=jrandom.PRNGKey(0)):
        
        y0, z0 = self.step.u_and_dudx(t=jnp.zeros((1, )), x=x0)
        
        # control = dt
        
        # term = diffrax.ControlTerm(self.step, control).to_ode()
        # TODO: add diffeqsolver support
        carry = (0, t0, dt, x0, y0, z0, key)

        def step_fn(carry, inp=None):
            return self.step(carry, inp)
        
        
        (carry, output) = jax.lax.scan(step_fn, carry, None, length=num_timesteps, unroll=unroll)
        return (carry, output)

@jax.jit
def sum_square_error(y, y_pred):
    """Computes the sum of square error."""
    return jnp.sum(jnp.square(y - y_pred))

def fetch_minibatch(rng, batch_size, num_timesteps, dim):  # Generate time + a Brownian motion
    T = 1.0
    M = batch_size
    N = num_timesteps
    D = dim

    dt = T / N * jnp.ones((M, 1))
    dW = jnp.sqrt(T / N) * jrandom.normal(rng, shape=(M, N, D))

    return dt, dW

def g_fn(X):
    return jnp.sum(X ** 2, axis=-1, keepdims=True)

def dg_fn(X):
    y, vjp_func = jax.vjp(g_fn, X)
    return vjp_func(jnp.ones(y.shape))[0]


@eqx.filter_jit
def train_step(model, x0, t0, dt, num_timesteps, optimizer, opt_state, unroll=1, key=jrandom.PRNGKey(0)):
    # batch_size = model.batch_size
    # t, W = data
    @eqx.filter_jit
    def loss_fn(model):
        loss = 0.0
        
        out_carry, out_val = jax.vmap(model, in_axes=(0, None, None, None, None, 0))(x0, t0, dt, num_timesteps, unroll, key)
        
        (_, _, _, x_final, y_final, z_final, _) = out_carry
        (x, y_tilde_list, y_list) = out_val
        
        loss += sum_square_error(y_tilde_list, y_list)
        loss += sum_square_error(y_final, g_fn(x_final))
        loss += sum_square_error(z_final, dg_fn(x_final))

        return (loss, y_list)

    (loss, y), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    # (loss, y), grads = grad_loss(model, x0, t0, dt, num_timesteps, unroll, key)
    
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    
    return loss, model, opt_state, y


def u_exact(t, X): # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return jnp.exp((r + sigma_max**2)*(1.0 - t))*jnp.sum(X**2, 1, keepdims = True)


def train(args):

    learning_rate = args.lr
    rng = jrandom.PRNGKey(0)

    model = NeuralFBSDE(in_size=args.dim + 1, out_size=1, width_size=args.width_size, depth=args.depth, noise_size=args.dim, key=rng, diffrax_solvers=args.diffrax_solvers)

    # train
    x0 = jnp.array([1.0, 0.5] * int(args.dim / 2))
    x0 = jnp.broadcast_to(x0, (args.batch_size, args.dim))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start = time.time()
    for step in range(args.num_iters):
        rng, _ = jrandom.split(rng)
        bm_key = jrandom.split(rng, args.batch_size)
        # data = fetch_minibatch(rng)
        loss, model, opt_state, y_pred = train_step(model, x0, 0.0, args.dt, args.num_timesteps, optimizer, opt_state, args.unroll, bm_key)

        if (step % args.print_every) == 0 or step == args.num_iters - 1:
            end = time.time()
            print(f"Step: {step}, Loss: {loss}, Computation time: "f"{end - start}")
    
    if args.plot:
        samples = fetch_minibatch(rng, args.batch_size, args.num_timesteps, args.dim)
        _, W_test = samples

        Dt = jnp.concatenate([jnp.zeros((args.batch_size, 1), dtype=jnp.float32), jnp.ones((args.batch_size, args.num_timesteps)) * 1.0 / args.num_timesteps], axis=1).reshape((args.batch_size, args.num_timesteps+1, 1))
        t_test = jnp.cumsum(Dt, axis=1)  # M x (N+1) x 1

        rng = jrandom.split(rng, args.batch_size)
        out_carry, out_val = jax.vmap(model, in_axes=(0, None, None, None, None, 0))(x0, 0.0, args.dt, args.num_timesteps, args.unroll, rng)
        _, (X_pred, y_tilde_list, Y_pred) = out_carry, out_val

        # X_pred = jnp.transpose(X_pred, (1, 0, 2))
        # print(X_pred.shape)
        # Y_pred = jnp.transpose(Y_pred, (1, 0, 2))

        Y_test = jnp.reshape(u_exact(jnp.reshape(t_test[0: args.batch_size, 0: args.num_timesteps, :],[-1, 1]), jnp.reshape(X_pred,[-1, args.dim])),[args.batch_size, -1, 1])

        n_samples = 3
            
        plt.figure()
        plt.plot(t_test[0:1,1:,0].T, Y_pred[0:1,:,0].T, 'b', label='Learned $u(t,X_t)$')
        plt.plot(t_test[0:1,1:,0].T, Y_test[0:1,:,0].T, 'r--', label='Exact $u(t,X_t)$')
        plt.plot(t_test[0:1,-1,0], Y_test[0:1,-1,0], 'ko', label='$Y_T = u(T,X_T)$')

        plt.plot(t_test[1:n_samples,1:,0].T, Y_pred[1:n_samples,:,0].T,'b')
        plt.plot(t_test[1:n_samples,1:,0].T,Y_test[1:n_samples,:,0].T,'r--')

        # print(Y_pred[1:n_samples,:,0].T)
        # print(Y_test[1:n_samples,:,0].T)
        # plt.plot(t_test[1:n_samples,1:,0],Y_test[1:n_samples,-1,0],'ko')

        plt.plot([0],Y_test[0,0,0],'ks',label='$Y_0 = u(0,X_0)$')

        plt.xlabel('$t$')
        plt.ylabel('$Y_t = u(t,X_t)$')
        plt.title('100-dimensional Black-Scholes-Barenblatt')
        plt.legend()
        plt.savefig("fbsde.png")
        plt.show()


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width-size', type=int, default=64)
    parser.add_argument('--num-timesteps', type=int, default=100)
    parser.add_argument('--num-iters', type=int, default=1000)
    parser.add_argument('--unroll', type=int, default=1)
    parser.add_argument('--print-every', type=int, default=200)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--diffrax-solver', type=bool, default=False)
    
    # test code
    
    args = parser.parse_args()

    # warm up run
    train(args)
    # unroll_list = [2, 5, 10, 15, 20, 30, 40, 50]
    # for unroll in unroll_list:
    #     args.unroll = unroll
    #     train(args)


if __name__ == '__main__':
    main()

