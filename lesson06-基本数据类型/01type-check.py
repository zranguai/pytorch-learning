import torch
import numpy as np
a = torch.randn(2, 3)    # 随机生成2行3列的矩阵

# print(a.shape)
# print(a.size(0))
# print(a.size(1)) # 返回shape的第2个元素
# print(a.shape[1])    # 3
# # print(a)
# cpu上
# print(a.type())    # torch.FloatTensor
# print(type(a))
# print(isinstance(a, torch.FloatTensor))
#
# Gpu上
# data = a.cuda()
# print(isinstance(data, torch.cuda.FloatTensor))

# 维度为0的标量的tensor：一般用来计算loss
# a = torch.tensor(1.)
# print(torch.tensor(1.))
# print(a.shape)
# print(len(a.shape))

# 维度为1的向量：一般用于bias/Linear input
# b = torch.tensor([1, 1])
# c = torch.tensor([1.1, 2.2])
# print(b)
# print(b.shape)
# print(c)
# print(torch.FloatTensor(3))    # 随机生成3个出来
#
# data = np.ones(2)
# print(torch.from_numpy(data))


"""
在pytorch0.3的版本中dimention为0的tensor是不存在的，如果表达是标量返回[0.3]
在之后的版本中，标量返回0.3 (为了语义更加清晰，使用长度为0的标量)

区分dim/size/shape/tensor
[2, 2]
dim: 2  rank
size/shape: [2, 2]  
tensor: 具体数字 [1, 3 ]
                 [2, 4]
"""
# Dim=2
# h = torch.tensor([2, 2])
# print(h.shape)
# print(h)
# print(h.size(0))    # 2


