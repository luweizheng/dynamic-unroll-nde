import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import equinox as eqx

# from jax.config import config
# We use GPU as the default backend.
# If you want to use cpu as backend, uncomment the following line.
# config.update("jax_platform_name", "cpu")

class MLP(eqx.Module):
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    dense3: eqx.nn.Linear
    
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = eqx.nn.Linear(in_features=16, out_features=256, key=jrandom.PRNGKey(0))
        self.dense2 = eqx.nn.Linear(in_features=256, out_features=256, key=jrandom.PRNGKey(0))
        self.dense3 = eqx.nn.Linear(in_features=256, out_features=16, key=jrandom.PRNGKey(0))

    def __call__(self, x):
        # 使用定义好的运算构建前向网络
        x = self.dense1(x)
        x = jnn.relu(x)
        x = self.dense2(x)
        x = jnn.relu(x)
        x = self.dense3(x)
        return x

class Con(eqx.Module):
    """
    """
    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    conv3: eqx.nn.Conv
    relu: jnn.relu

    def __init__(self):
        super(Con, self).__init__()
        self.conv1 = eqx.nn.Conv2d(in_channels=16, out_channels=256, kernel_size=3, padding=1, key=jrandom.PRNGKey(0))
        self.conv2 = eqx.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, key=jrandom.PRNGKey(0))
        self.conv3 = eqx.nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding=1, key=jrandom.PRNGKey(0))
        self.relu = jnn.relu

    def __call__(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

class Unroll(eqx.Module):
    mlp: MLP
    unroll: int

    def __init__(self):
        super(Unroll, self).__init__()
        self.mlp = Con()
        self.unroll = 20
    
    def __call__(self, x, num_steps):
        def step_fn(x, input=None):
            y = self.mlp(x)

            return y, y
        
        x, _ = jax.lax.scan(step_fn, x, xs=None, length=num_steps, unroll=self.unroll)
        
        return x


x = jnp.ones(shape=(128, 16, 224, 224))
model = Unroll()
num_steps = 100

@eqx.filter_jit
def forward_fn(x):
    y = jax.vmap(model, in_axes=(0, None))(x, num_steps)
    return y

start = time.time()
y_pred = forward_fn(x)
end = time.time()
print(f"time: {end - start}")

start = time.time()
y_pred = forward_fn(x)
end = time.time()
print(f"time: {end - start}")

# z=jax.xla_computation(forward_fn)(x)

# with open("steps8_not_unroll.dot", "w") as f:
#     f.write(z.as_hlo_dot_graph())
# end = time.time()
# print(f"time: {end - start}")
