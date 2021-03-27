"""
计算准确度的代码
"""

import torch
from torch.nn import functional as F
logits = torch.rand(4, 10)

pred = F.softmax(logits, dim=1)
print(pred)

pred_label = pred.argmax(dim=1)    # 取最大值的下标

print(pred_label)

label = torch.tensor([9, 3, 2, 9])
correct = torch.eq(pred_label, label)
print(correct)
print(correct.sum().float().item()/4)    # item()作用是得到里面的元素
