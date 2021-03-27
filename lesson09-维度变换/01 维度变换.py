import torch
"""
1. view    # 将一个shape转换成例一个shape
2. squeeze(减少维度)/unsqueeze(增加维度)
3. transpose(单维变换)/t/repeat(多维变换)
4. expand(改变理解方式)/repeat(实实在在增加数据 memory copied)
"""

# view: lost dim information
a = torch.rand(4, 1, 28, 28)
print(a)
print(a.shape)
print(a.view(4, 28 * 28).shape)
print(a.view(4 * 28, 28).shape)
print(a.view(4*1, 28, 28).shape)
b = a.view(4, 784)
b.view(4, 28, 28, 1)    # logic bug

# flexible but prone to corrupt
print(a.view(4, 783))    # RuntimeError: shape '[4, 783]' is invalid for input of size 3136



# squeeze / unsqueeze
"""
范围:
    [-a.dim()-1, a.dim()+1]
    [-5, 5)
"""
a = torch.rand(4, 1, 28, 28)
print(a.shape)
print(a.unsqueeze(0).shape)
print(a.unsqueeze(-1).shape)
print(a.unsqueeze(4).shape)
print(a.unsqueeze(-5).shape)
print(a.unsqueeze(5).shape)    # IndexError: Dimension out of range (expected to be in range of [-5, 4], but got 5)

a = torch.tensor([1.2, 2.3])
print(a)
print(a.unsqueeze(-1))
print(a.unsqueeze(0))

# 案例:
b = torch.rand(32)
f = torch.rand(4, 32, 14, 14)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape)

# squeeze
b = torch.rand(1, 32, 1, 1)
print(b.squeeze())    # 能压缩的都压缩
print(b.squeeze(0).shape)    # 压缩第0个元素
print(b.squeeze(-1).shape)
print(b.squeeze(1).shape)    # 32不能压缩就不压缩
print(b.squeeze(-4).shape)



# expand/repeat
# expand: broadcasting  改变理解方式
# repeat: memory copied  实实在在的增加数据
a = torch.rand(4, 32, 14, 14)

b = torch.rand(1, 32, 1, 1)
print(b)
print(b.expand(4, 32, 14, 14))    # torch.Size([4, 32, 14, 14])

print(b.expand(-1, 32, -1, -1).shape)    # -1表示该维度不变
print(b.expand(-1, 32, -1, -4).shape)    # 写-4变-4    RuntimeError: invalid shape dimension -128


# repeat:不建议使用
print(b.repeat(4, 32, 1, 1).shape)    # 第二个拷贝32次
print(b.repeat(4, 1, 1, 1).shape)
print(b.repeat(4, 1, 32, 32).shape)    #


# t():转置 只适合2D tensor
a = torch.randn(3, 4)
print(a)
print(a.t())


# Transpose: 维度变换
a = torch.rand(4, 3, 32, 32)
print(a.shape)
"""
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
"""
a1 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 3, 32, 32)    # 要加contigous
a2 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 3, 32, 32).transpose(1, 3)
print(a1.shape)
print(a2.shape)




# permute:可以直接排位置，可以使用任意多次的transpose来达到他的目的
a = torch.rand(4, 3, 28, 28)
print(a.transpose(1, 3).shape)    # torch.Size([4, 28, 28, 3])
b = torch.rand(4, 3, 28, 32)
print(b.transpose(1, 3).shape)    # torch.Size([4, 32, 28, 3])
print(b.transpose(1, 3).transpose(1, 3).shape)    # torch.Size([4, 3, 28, 32])
print(b.permute(0, 2, 3, 1).shape)    # torch.Size([4, 28, 32, 3])
























