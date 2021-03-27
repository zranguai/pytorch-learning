import torch
import torch.nn as nn
from torch.nn import functional as F

x = torch.randn(1, 10)
print(x.shape)

# 调用方式:类方法
class ML(nn.module):
    def __init__(self):
        super(ML, self).__init__()

        self.model = nn.Sequential(  # 构建模型
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )
# 函数方法
x = F.relu(x, inplace=True)











