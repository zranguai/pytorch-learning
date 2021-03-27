import torch

# indexing
# a = torch.rand(4, 3, 28, 28)
# print(a[0])
# print(a[0].shape)    # torch.Size([3, 28, 28]) ：索引第一个维度 ：取第0张图片
#
# print(a[0, 0].shape)    # torch.Size([28, 28])：第二个维度：第0张图片的第0个通道
#
# print(a[0, 0, 2])
# print(a[0, 0, 2, 4])    # tensor(0.9441) : 第0张图片第0个通道第二行第4列



# # select first/last N
# a = torch.rand(4, 3, 28, 28)
# print(a.shape)    # torch.Size([4, 3, 28, 28])
# print(a[:2].shape)    # torch.Size([2, 3, 28, 28])
# print(a[:2, :1, :, :].shape)    # torch.Size([2, 1, 28, 28])
# print(a[:2, 1:, :, :].shape)    # torch.Size([2, 2, 28, 28])
# print(a[:2, -1:, :, :].shape)    # torch.Size([2, 1, 28, 28])



# # select by steps
# a = torch.rand(4, 3, 28, 28)
# print(a[:, :, 0:28:2, 0:28:2].shape)    # torch.Size([4, 3, 14, 14])
#
# print(a[:, :, ::2, ::2].shape)    # torch.Size([4, 3, 14, 14])



# # select by specific index
# a = torch.rand(4, 3, 28, 28)
# print(a)
# print(a.index_select(0, torch.tensor([0, 2])).shape)    # 第1个参数的第0和第1个
# print(a.index_select(2, torch.arange(20)).shape)

# # ... 任意多的维度
# print(a[...].shape)    # torch.Size([4, 3, 28, 28])
# print(a[:, 1, ...].shape)    # torch.Size([4, 28, 28])
# print(a[..., :2].shape)    # torch.Size([4, 3, 28, 2])



# # select by mask
# x = torch.randn(3, 4)
# y = torch.randn(3, 4)
# print(x)
# mask = x.ge(0.5)    # >=0.5的位置为True
# print(mask)
# print(torch.masked_select(y, mask))    # 为True的位置选出来



# select by flatten index
src = torch.IntTensor(3, 4)
print(src)
print(torch.take(src, torch.tensor([0, 2, 5])))














