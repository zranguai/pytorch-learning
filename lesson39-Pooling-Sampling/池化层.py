
"""
outline:
    Pooling
    upsample
    Relu
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
x = torch.randn(1, 16, 14, 14)
print(x.shape)    # torch.Size([1, 16, 14, 14])

# 从nn中导入最大池化
layer = nn.MaxPool2d(2, stride=2)
out = layer(x)
print(out.shape)    # torch.Size([1, 16, 7, 7])    (14-2)/2 + 1 = 7


# 使用F.的方式平均池化
out = F.avg_pool2d(x, 2, stride=2)    # torch.Size([1, 16, 7, 7])
print(out.shape)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# upsample
# 采用F.interpolate
# interpolate: 是插值的意思
# +++++++++++++++++++++++++++++++++++++++++++++++++++++#
x = out
out = F.interpolate(x, scale_factor=2, mode='nearest')    # 采用最近邻采样
print(out.shape)    # torch.Size([1, 16, 14, 14])

out = F.interpolate(x, scale_factor=3, mode='nearest')
print(out.shape)    # torch.Size([1, 16, 21, 21])

#------------------------------------------------#
#  Relu激活函数
#
# ------------------------------------------------#

x = torch.randn(1, 16, 7, 7)
print(x.shape)    # torch.Size([1, 16, 7, 7])

# 方法1:采用nn.的方式
layer = nn.ReLU(inplace=True)    # inplace=True x--->x'(x'使用x内存空间)
out = layer(x)
print(out.shape)    # torch.Size([1, 16, 7, 7])

# 方法2：采用F.的方式
out = F.relu(x)
print(out.shape)    # torch.Size([1, 16, 7, 7])




