import torch
from torch.nn import functional as F
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)
print(x)
print(w)
o = torch.sigmoid(x@w.t())    # 这里没有写bias
print(o)
print(torch.ones(1, 1))
loss = F.mse_loss(torch.ones(1, 1), o)
print(loss)
loss.backward()
print(w.grad)



















