import time
import mindspore.numpy as mnp
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as P
import mindspore.dataset as ds
import mindspore.common.dtype as mstype

class MLP(nn.Cell):
    """
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = nn.Dense(in_channels=64, out_channels=64)
        self.relu = nn.ReLU()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.dense1(x)
        x = self.relu(x)
        return x


class Conv(nn.Cell):
    """
    """
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, pad_mode="same")
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, pad_mode="same")
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, pad_mode="same")
        self.relu = nn.ReLU()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class Unroll(nn.Cell):
    def __init__(self):
        super(Unroll, self).__init__()
        self.mlp = MLP()
        self.unroll = 250
    
    def construct(self, x, num_steps):
        i = P.ScalarToTensor()(0, mstype.int64)
        steps = num_steps / self.unroll
        while i < steps:
            for j in range(self.unroll):
                x = self.mlp(x)
            i += 1
        
        return x

network = MLP()

x = mnp.ones(shape=(128, 64))

num_steps = mnp.asarray(500)

context.set_context(
        mode=context.GRAPH_MODE,
        device_target="GPU",
        device_id=0,
        save_graphs=False
)

model = Unroll()

start = time.time()
y_pred = model(x, num_steps)
end = time.time()
print(f"time: {end - start}")

start = time.time()
y_pred = model(x, num_steps)
end = time.time()
print(f"time: {end - start}")