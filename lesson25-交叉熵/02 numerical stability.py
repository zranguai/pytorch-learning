import torch
from torch.nn import functional as F
x = torch.randn(1, 784)
w = torch.randn(10, 784)
logits = x@w.t()
print(logits.shape)
pred = F.softmax(logits, dim=1)
print(pred)
pred_log = torch.log(pred)
loss1 = F.nll_loss(pred_log, torch.tensor([3]))
print(loss1)
loss2 = F.cross_entropy(logits, torch.tensor([3]))    # 这里使用logits, 因为cross_entropy = softmax + log + nll_loss   (这三个操作一起)
print(loss2)















