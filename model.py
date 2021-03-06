#%%
from TCN.TCN.tcn import TemporalConvNet
import torch
from torch import nn
from torch import Tensor
import auraloss
from utils import tone

#%%
class Buffer(nn.Module):
    def __init__(self):
        super(Buffer, self).__init__()
        self.reset()

    def reset(self):
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

        # it's the -1th element, right, which is temporally dependant on the history?
        output = self.linear(output[:, :, -1]).squeeze()
        # print(output.shape)

        self.buffer.push(output)
        return output



# %% tests
model = TCN()

pre_shape = model.buffer.data.shape
model()
# print(model.buffer.data.shape)
assert model.buffer.data.shape == pre_shape
model()
# print(model.buffer.data.shape)
assert model.buffer.data.shape == pre_shape
del(model)

#%%
import matplotlib.pyplot as plt
# %%
def train(model, epochs, y, lossfn, optimizer):
    model.train()
    for i in range(epochs):
        print(f'Epoch {i} ', end=None)
        samples = len(y)
        outputs = torch.zeros((samples))
        for i in range(samples):
            outputs[i] = model()

        loss = lossfn(outputs.view((1,-1)), y.view((1,-1)))
        print(f'- loss: {loss}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.buffer.reset()

        plt.plot(outputs.detach())
        plt.show()

    return outputs

#%%
model = TCN()
y = Tensor(tone(220, sr=48000, length=1024))
print(y.view((1,-1)).shape)
lossfn = auraloss.freq.STFTLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


#%%
train(model, 5, y, lossfn, optimizer)

# %%
