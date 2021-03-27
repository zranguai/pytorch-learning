import torch
import torch.nn as nn

# ----------------------------#
# BatchNorm1d
# ----------------------------#
x = torch.randn(100, 16) + 0.5
print(x.shape)

layer = torch.nn.BatchNorm1d(16)    # 这个必须与前面的匹配起来否则会报错

print(layer.running_mean, layer.running_var)
"""
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) 
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
"""
out = layer(x)
print(layer.running_mean, layer.running_var)
"""
tensor([0.0452, 0.0446, 0.0516, 0.0671, 0.0644, 0.0622, 0.0514, 0.0449, 0.0520,
        0.0546, 0.0461, 0.0620, 0.0332, 0.0450, 0.0384, 0.0580]) 
tensor([0.9868, 0.9935, 1.0214, 1.0137, 1.0009, 0.9895, 1.0065, 1.0319, 0.9841,
        1.0051, 0.9967, 0.9968, 1.0045, 0.9877, 1.0011, 1.0031])
"""
#----------------------------------------#
# 这里的分布服从于 U(0.5, 1)
#
# ---------------------------------------#

x = torch.randn(100, 16) + 0.5
layer = torch.nn.BatchNorm1d(16)

for i in range(5):    # 疑问？？？？？？？？，每循环一次经过一次batchnorm
    out = layer(x)

print(layer.running_mean, layer.running_var)


# ---------------------------#
# nn.BatchNorm2d
# ---------------------------#
x = torch.rand(1, 16, 7, 7)
print(x.shape)

layer = nn.BatchNorm2d(16)
out = layer(x)
print(out.shape)    # torch.Size([1, 16, 7, 7])

print(layer.weight)
"""
这里的weight,bias更权重的那个不太一样
"""
print(layer.weight.shape)    # torch.Size([16])

print(layer.bias.shape)    # torch.Size([16])

# -----------------------------------#
#  class variables
# -----------------------------------#
print(vars(layer))


# ------------------------------------#
# Test
# ------------------------------------#
layer.eval()    # 加这行表示现在是在test阶段
out = layer(x)
print(vars(layer))

























