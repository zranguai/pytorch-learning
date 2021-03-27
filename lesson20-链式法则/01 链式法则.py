import torch

x = torch.tensor(1.)
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2., requires_grad=True)
b2 = torch.tensor(1.)

y1 = x*w1 + b1
y2 = y1*w2 + b2

dy2_dy1 = torch.autograd.grad(y2, [y1], retain_graph=True)[0]
dy1_dw1 = torch.autograd.grad(y1, [w1], retain_graph=True)[0]

dy2_dw1 = torch.autograd.grad(y2, w1, retain_graph=True)[0]    # 这里的w1加不加[]都行？？

print(dy2_dy1*dy1_dw1)

print(dy2_dw1)























