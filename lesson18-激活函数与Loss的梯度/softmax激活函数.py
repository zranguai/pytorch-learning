"""
softmax求导:
    pi(1-pj)    if i=j
    -pj*pi      if i!=j

    1  if i=j
    0  if i!=j
"""
import torch
from torch.nn import functional as F
a = torch.rand(3)    # tensor([0.4207, 0.2955, 0.8440])
print(a.requires_grad_())    # 这样之后就可以求梯度 tensor([0.5424, 0.1913, 0.9416], requires_grad=True)

p = F.softmax(a, dim=0)    # 自动完成建图操作 tensor([0.2489, 0.3556, 0.3954], grad_fn=<SoftmaxBackward>)

# 当你调用一次backward之后除了完成一次反向传播以外，还会把这个图的梯度信息清除掉

print(torch.autograd.grad(p[1], [a],retain_graph=True))    # (tensor([-0.0755,  0.1879, -0.1125]),)    i=1为正的其他的为负的
# #
# #
print(torch.autograd.grad(p[2], [a]))    # (tensor([-0.1349, -0.1125,  0.2473]),)    # i=2为正的其他的为负的









