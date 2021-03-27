import torch


# 激活函数
z = torch.linspace(-100, 100, 10)
# sigmoid激活函数
print(z)
print(torch.sigmoid(z))    # 范围在0-1

# tanh激活函数: 在rnn中用的比较多 取值范围为-1-1
a = torch.linspace(-1, 1, 10)
print(torch.tanh(a))

# Relu激活函数
# 在pytorch中的两种实现：1.从torch.nn中  2. 从torch.relu中
from torch.nn import functional as F
a = torch.linspace(-1, 1, 10)
print(torch.relu(a))
print(F.relu(a))






