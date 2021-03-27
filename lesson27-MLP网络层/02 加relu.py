import torch
import  torch.nn as nn
from torch.nn import functional as F

x = torch.randn(1, 784)    # torch.Size([1, 784])
print(x.shape)

layer1 = nn.Linear(784, 200)    # 在这里第一个参数使ch-in 第二个参数是ch-out
layer2 = nn.Linear(200, 200)
layer3 = nn.Linear(200, 10)

x = layer1(x)
x = F.relu(x, inplace=True)    # 表示是否需要覆盖之前的值，inplace=True表示需要覆盖。为了节省内存
print(x.shape)    # torch.Size([1, 200])

x = layer2(x)
x = F.relu(x, inplace=True)
print(x.shape)    # torch.Size([1, 200])

x = layer3(x)
x = F.relu(x, inplace=True)
print(x.shape)    # torch.Size([1, 10])
print(x)



