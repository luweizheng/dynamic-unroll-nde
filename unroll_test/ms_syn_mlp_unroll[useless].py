import time
import mindspore.numpy as mnp
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as P
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
from dataclasses import dataclass

@dataclass
class Args:
    batch_size: int

    # dim of SDE
    hidden_size: int
    noise_size: int 
    num_timesteps: int
    num_iters: int
    
    # network
    mu_depth: int
    mu_width_size: int
    sigma_depth: int
    sigma_width_size: int
    
    # dynamic unroll
    unroll: int
    T: float = 1.0
    
class MuField(nn.Cell):

    def __init__(self, hidden_size, width_size, **kwargs):
        super().__init__(**kwargs)
        self.d1 = nn.Dense(in_channels=hidden_size + 1, out_channels=width_size)
        self.d2 = nn.Dense(in_channels=width_size, out_channels=width_size)
        self.d3 = nn.Dense(in_channels=width_size, out_channels=width_size)
        self.d4 = nn.Dense(in_channels=width_size, out_channels=hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def construct(self, t, y):
        x = mnp.concatenate([t, y], axis=1)
        x = self.d1(x)
        x = self.relu(x)
        x = self.d2(x)
        x = self.relu(x)
        x = self.d3(x)
        x = self.relu(x)
        x = self.d4(x)
        x = self.tanh(x)
        return x

@ms.ms_function
def lipswish(x):
    return 0.909 * x * nn.LogSigmoid()(x)

class SigmaField(nn.Cell):
    noise_size: int
    hidden_size: int

    def __init__(
        self, noise_size, hidden_size, width_size,  **kwargs):
        super().__init__(**kwargs)
        self.d1 = nn.Dense(in_channels=hidden_size + 1, out_channels=width_size)
        self.d2 = nn.Dense(in_channels=width_size, out_channels=width_size)
        self.d3 = nn.Dense(in_channels=width_size, out_channels=width_size)
        self.d4 = nn.Dense(in_channels=width_size, out_channels=hidden_size)
        self.lipswish = lipswish
        self.noise_size = noise_size
        self.hidden_size = hidden_size

    def __call__(self, t, y):
        x = mnp.concatenate([t, y], axis=1)
        x = self.d1(x)
        x = self.lipswish(x)
        x = self.d2(x)
        x = self.lipswish(x)
        x = self.d3(x)
        x = self.lipswish(x)
        x = self.d4(x)
        x = self.tanh(x)
        return x.reshape(self.noise_size, self.hidden_size)


class SDEStep(nn.Cell):
    mf: MuField  # drift
    sf: SigmaField  # diffusion
    noise_size: int

    def __init__(
        self,
        noise_size,
        hidden_size,
        mu_width_size,
        sigma_width_size,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mf = MuField(hidden_size, mu_width_size)
        self.sf = SigmaField(
            noise_size, hidden_size, sigma_width_size)

        self.noise_size = noise_size

    def construct(self, carry):
        (i, t0, dt, y) = carry
        t = t0 + i * dt
        bm = mnp.randn((self.noise_size, )) * mnp.sqrt(dt)
        drift_term = self.mf(t=t, y=y) * dt
        diffusion_term = mnp.dot(self.sf(t=t, y=y0), bm)
        y = y + drift_term + diffusion_term
        carry = (i+1, t0, dt, y)

        return carry, y


class NeuralSDE(nn.Cell):
    step: SDEStep
    noise_size: int
    hidden_size: int
    mu_depth: int
    sigma_depth: int
    mu_width_size: int
    sigma_width_size: int


    def __init__(
        self,
        noise_size,
        hidden_size,
        mu_width_size,
        sigma_width_size,
        **kwargs,
    ):
        super().__init__(**kwargs)


        self.step = SDEStep(noise_size=noise_size,
            hidden_size=hidden_size,
            mu_width_size=mu_width_size,
            sigma_width_size=sigma_width_size)

        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.mu_width_size = mu_width_size
        self.sigma_width_size = sigma_width_size
    
    def construct(self, y0, t0, dt, num_timesteps, unroll):

        # batch_size, _  = y0.shape
        i = P.ScalarToTensor()(0, mstype.int64)
        rem = num_timesteps % unroll
        steps = num_timesteps // unroll
        carry = (0, t0, dt, y0)
        while i < steps:
            for j in range(unroll):
                carry, _ = self.step(carry)
            i += 1
        
        while i < rem:
            carry, _ = self.step(carry)
            i += 1
            
        return carry




y0 = mnp.ones(shape=(128, 64))

num_steps = mnp.asarray(500)

context.set_context(
        mode=context.GRAPH_MODE,
        device_target="GPU",
        device_id=0,
        save_graphs=False
)

model = NeuralSDE(64, 64,64,64)

t0 = mnp.zeros(shape=(128, 1))

_, _, _, ans = model(y0, t0, 0.1, num_steps, unroll=1)

print(ans)


# start = time.time()
# y_pred = model(x, num_steps)
# end = time.time()
# print(f"time: {end - start}")

# start = time.time()
# y_pred = model(x, num_steps)
# end = time.time()
# print(f"time: {end - start}")