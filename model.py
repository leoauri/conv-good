#%%
from TCN.TCN.tcn import TemporalConvNet
import torch
from torch import nn
from torch import Tensor

#%%
class Buffer(nn.Module):
    def __init__(self):
        self.data = torch.zeros(128)

    def push(self, x):
        self.data = self.data.roll(1, 0)
        self.data[0] = x


#%% tests
buffer = Buffer()

pre_shape = buffer.data.shape
buffer.push(Tensor([1.45]))

assert buffer.data.shape == pre_shape
assert buffer.data[0] == 1.45

buffer.push(Tensor([0.92]))

assert buffer.data[1] == 1.45
assert buffer.data[0] == 0.92
assert buffer.data.shape == pre_shape


#%%
class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(1, [128] * 4, 5, dropout=0.25)
        self.linear = nn.Linear(128, 1)
        self.buffer = Buffer()

    def forward(self):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(self.buffer.data.view(1,1,-1))

        output = self.linear(output[:, :, -1]).squeeze()
        print(output.shape)

        self.buffer.push(output)
        return output


model = TCN()




# %%
pre_shape = model.buffer.data.shape
model()
# print(model.buffer.data.shape)
assert model.buffer.data.shape == pre_shape
model()
# print(model.buffer.data.shape)
assert model.buffer.data.shape == pre_shape


# %%
