import torch
"""
Merge or split
合并:
cat
stack
分割:
split
chunk
"""
# # cat
# a = torch.rand(4, 32, 8)
# b = torch.rand(5, 32, 8)
# print(torch.cat([a, b], dim=0).shape)    # torch.Size([9, 32, 8])


# a1 = torch.rand(4, 3, 32, 32)
# a2 = torch.rand(4, 1, 32, 32)
# print(torch.cat([a1, a2], dim=0).shape)    # RuntimeError: invalid argument 0, 其他维度要一致
# print(torch.cat([a1, a2], dim=1).shape)    # torch.Size([4, 4, 32, 32])

# a1 = torch.rand(4, 3, 16, 32)
# a2 = torch.rand(4, 3, 16, 32)
#
# print(torch.cat([a1, a2], dim=2).shape)    # torch.Size([4, 3, 32, 32])



# # stack: create a new dim: 需求 维度完全一致
# a1 = torch.rand(4, 3, 16, 32)
# a2 = torch.rand(4, 3, 16, 32)
# print(torch.cat([a1, a2], dim=2).shape)    # torch.Size([4, 3, 32, 32])
# print(torch.stack([a1, a2], dim=2).shape)    # torch.Size([4, 3, 2, 16, 32])
# a = torch.rand(32, 8)
# b = torch.rand(32, 8)
# print(torch.stack([a, b], dim=0).shape)    # torch.Size([2, 32, 8])



# split: by len: 根据长度来分
b = torch.rand(32, 8)
a = torch.rand(32, 8)
# print(a.shape)    # torch.Size([32, 8])
c = torch.stack([a, b], dim=0)
# print(c.shape)    # torch.Size([2, 32, 8])
aa, bb = c.split([4, 4], dim=2)
print(aa.shape, bb.shape)    # torch.Size([1, 32, 8]) torch.Size([1, 32, 8])

# aa, bb = c.split(2, dim=0)    # ValueError: not enough values to unpack (expected 2, got 1)

print(c.shape)    # torch.Size([2, 32, 8])
# chunk: by num: 根据数量来分
aa, bb = c.chunk(2, dim=2)    # torch.Size([2, 32, 8])
print(aa.shape, bb.shape)    # torch.Size([1, 32, 8]) torch.Size([1, 32, 8])








