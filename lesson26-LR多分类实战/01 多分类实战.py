import torch
from torch.nn import functional as F
w1, b1 = torch.randn(200, 784, requires_grad=True), torch.randn(200, requires_grad=True)    # randn里面参数是ch-out  ch-in
w2, b2 = torch.randn(200, 200, requires_grad=True), torch.randn(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), torch.randn(10, requires_grad=True)


def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x


































