import torch
"""
Math operation
1. add/minus/multiply/divide
2. matmul
3. pow
4. sqrt/rsqrt
5. round
"""
# 基础部分
# a = torch.rand(3, 4)
# b = torch.rand(4)
#
# print(a)
# print(b)
# print(a + b)    # b会被广播
# # all()函数的功能: 如果张量tensor中所有元素都是True, 才返回True; 否则返回False
# b = torch.tensor([1, 1, 1, 1])
# print(torch.all(torch.eq(a-b, torch.sub(a, b))))


# # matmul
# # torch.mm
# #     only for 2d
# # torch.matmul
# # @
# a = torch.tensor([[3., 3.],
#                  [3., 3.]])
# print(a)
# b = torch.ones(2, 2)
# print(b)
#
# print(torch.mm(a, b))    # 只针对2d矩阵
#
# print(torch.matmul(a, b))
#
# print(a@b)
#
# # 案例:
# # ==2d的tensor运算
# a = torch.rand(4, 784)
# x = torch.rand(4, 784)
# w = torch.rand(512, 784)    # 分别为ch-out ch-in
#
# print((x.@w.t()).shape)    # torch.Size([4, 512]) ×时第一个元素为out，所以需要转置
#
# print(torch.matmul(x, w.t()).shape)    # torch.Size([4, 512])
#
# # >2d的tensor运算
# a = torch.rand(4, 3, 28, 64)
# b = torch.rand(4, 3, 64, 32)
# print(torch.matmul(a, b).shape)    # torch.Size([4, 3, 28, 32])
# b = torch.rand(4, 1, 64, 32)
# print(torch.matmul(a, b).shape)    # torch.Size([4, 3, 28, 32]), 这种情况会先使用broadcast,再使用矩阵相乘



# power
# a = torch.full([2, 2], 3)
# print(a.pow(2))
# print(a**2)
# aa = a**2
# print(aa.sqrt())
# print(aa.rsqrt())
# print(aa**(0.5))


# exp/log
a = torch.exp(torch.ones(2, 2))
print(a)
print(torch.log(a))



# # approximation
# a = torch.tensor(3.14)
# print(a.floor(), a.ceil(), a.trunc(), a.frac())    # tensor(3.) tensor(4.) tensor(3.) tensor(0.1400)
# #      往下取整    往上取整   截取,保留整数  截取,保留小数
#
# a = torch.tensor(3.499)
# print(a.round())    # tensor(3.)  四舍五入
# a = torch.tensor(3.5)
# print(a.round())    # tensor(4.)



# clamp：裁剪
"""
gradient clipping
(min)
(min, max)
"""
grad = torch.rand(2, 3)*15
print(grad)
print(grad.max())
print(grad.median())
print(grad.clamp(10))    # 里面的元素小于10的全部变成10
print(grad.clamp(2, 10))    # 小于2的裁剪成2， 大于10的裁剪成10





