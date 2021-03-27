"""
tensor 高级操作
where
gather: 收集，gather语句类似于查表的过程.   设计目的:使用GPU实现CPU的功能
"""
import torch
# where
# torch.where(condition,x,y) --> Tensor
# 案例:
# cond = torch.tensor([[0.6769, 0.7271],
#                     [0.8884, 0.4163]])
# print(cond)
# a = torch.zeros(2, 2)
# print(a)
# b = torch.ones(2, 2)
# print(b)
#
# print(torch.where(cond > 0.5, a, b))    # 如果cond成立，选取a中的元素，否则选择b中的元素


# gather
# torch.gather(input, dim, index, out=None) --> Tensor
"""
Gathers values along an axis specified by dim
for a 3-d tensor the output is specified by
out[i][j][k] = input[index[i][j][k][j][k]]    # if dim==0
out[i][j][k] = input[i][index[i][j][k][k]]    # if dim==1
out[i][j][k] = input[i][j][index[i][j][k]]    # if dim==2
"""
"""
retrieve global label
argmax (pred) to get relative labeling
on some condition, our label is dinstinct from relative labeling
"""


# 案例:检索  retrieve label
prob = torch.randn(4, 10)
# print(prob)

idx = prob.topk(dim=1, k=3)
# print(idx)
idx = idx[1]
# print(idx)
label = torch.arange(10) + 100
# print(label)
label_expand = label.expand(4, 10)
print(label_expand)
print(idx)    # 这是索引
print('------------------')
# print(idx.long())    # 转换成Longtensor数据格式
print(torch.gather(label_expand, dim=1, index=idx.long()))    # 按照index索引进行取数据






















