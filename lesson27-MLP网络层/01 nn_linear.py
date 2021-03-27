import torch
import  torch.nn as nn

x = torch.randn(1, 784)    # torch.Size([1, 784])
print(x.shape)

layer1 = nn.Linear(784, 200)    # 在这里第一个参数使ch-in 第二个参数是ch-out
layer2 = nn.Linear(200, 200)
layer3 = nn.Linear(200, 10)

x = layer1(x)
print(x.shape)    # torch.Size([1, 200])

x = layer2(x)
print(x.shape)    # torch.Size([1, 200])

x = layer3(x)
print(x.shape)    # torch.Size([1, 10])
print(x)

















