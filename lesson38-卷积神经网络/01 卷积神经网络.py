import torch.nn as nn
import torch
from torch.nn import functional as F

# 第一个参数为input的chanel,第二个参数为kernel的数量，kernel_size=3*3    [1, 3, 26, 26]
layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)

out = layer.forward(x)
print(out.shape)    # torch.Size([1, 3, 26, 26])    # 26 = (28-3)/1 + 1


layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer.forward(x)
print(out.shape)    # torch.Size([1, 3, 28, 28])

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
out = layer.forward(x)
print(out.shape)    # torch.Size([1, 3, 14, 14])

out = layer(x)    # 会自动进行,运用了python的魔术方法 __call__
print(out.shape)    # torch.Size([1, 3, 14, 14])

print(layer.weight)    # 查看layer的权重
print(layer.weight.shape)    # torch.Size([3, 1, 3, 3])

print(layer.bias.shape)    # torch.Size([3])


# F.conv2D

# 上面x = torch.rand(1, 1, 28, 28)
w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)

# out = F.conv2d(x, w, b, stride=1, padding=1)
# print(out)    # 报错，一位x和w的chanels数对应不上
"""
RuntimeError: Given groups=1, weight of size 16 3 5 5, expected input[1, 1, 28, 28] to have 3 channels,
 but got 1 channels instead
"""
x = torch.randn(1, 3, 28, 28)
out = F.conv2d(x, w, b, stride=1, padding=1)
print(out.shape)    # torch.Size([1, 16, 26, 26])

out = F.conv2d(x, w, b, stride=2, padding=2)
print(out.shape)    # torch.Size([1, 16, 14, 14])





















