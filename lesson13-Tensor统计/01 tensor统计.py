"""
statistics
norm：范数
mean sum
prod
max, min, argmin(最小值的位置), argmax
kthvalue(第几个值 默认是小的: 比如第8个小的), topk(top多少)
"""
import torch
# # norm:
# a = torch.full([8], 1)
# b = a.view(2, 4)
# c = a.view(2, 2, 2)
# print(a)
# print(b)
# print(c)
# print(a.norm(1), b.norm(1), c.norm(1))    # nsor(8.) tensor(8.) tensor(8.)
# print(a.norm(2), b.norm(2), c.norm(2))    # tensor(2.8284) tensor(2.8284) tensor(2.8284)
#
# print(b.norm(1, dim=1))    # dim=1：将dim=1的部分取范数，同时二维向量变成一维向量  tensor([4., 4.])
# print(b.norm(2, dim=1))    # tensor([2., 2.])
#
# print(c.norm(1, dim=0))
# print(c.norm(2, dim=0))



# # mean sum min max prod(阶乘)
# a = torch.arange(8).view(2, 4).float()
# print(a)
# """
# tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]])
# """
# print(a.min(), a.max(), a.mean(), a.prod())    # tensor(0.) tensor(7.) tensor(3.5000) tensor(0.)
# print(a.sum())    # tensor(28.)
# print(a.argmin(), a.argmax())    # tensor(0) tensor(7)


# # argmin/argmax在指定维度的表示
# a = torch.rand(4, 5)
# print(a)
# print(a.argmax())
# print(a.argmax(dim=1))    # 在dim=1即取每个维度中最大值的位置



# # keepdim
# a = torch.rand(4, 10)
# print(a)
# # print(a.max(dim=1))
# print(a.argmax(dim=1))
# print(a.max(dim=1, keepdim=True))    # 这个会返回他在dim=1的最大值和最大值的位置



# top-k or k-th
# a = torch.rand(4, 10)
# print(a.topk(3, dim=1))
# print(a.topk(3, dim=1, largest=False))
#
# print(a.kthvalue(8, dim=1))    # 返回dim=1的第8大的值
"""
torch.return_types.kthvalue(
values=tensor([0.7363, 0.8011, 0.6856, 0.6297]),
indices=tensor([4, 0, 7, 8]))
"""



# compare
"""
>  >= <  <=  !=  ==

torch.eq(a, b)
"""
a = torch.rand(4, 10)
print(a > 5)    # 里面的每个元素都要比较
print(torch.gt(a, 0))
print(a != 0)

a = torch.ones(2, 3)
b = torch.randn(2, 3)

"""
疑问: torch.rand()和torch.randn()的区分?
答：rand()是均匀分布，randn()是标准正太分布
"""
print(a)
print(b)
print(torch.eq(a,b))

print(torch.eq(a, a))    # 返回每个元素
"""
tensor([[True, True, True],
        [True, True, True]])
"""
print(torch.equal(a, a))     # True    所有都为True才为True





