import torch
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image


# Dim=0,用于loss
a = torch.tensor(2.2)
print(a.shape)    # torch.Size([])
print(len(a.shape))    # 0
print(a.size())    # torch.Size([])

# Dim=1,用于Bias/Linear input
b = torch.tensor([2])    # 直接这样写，里面的数据类型跟着里面数据变化
print(b)
print(b.type())
c = torch.tensor([1.1, 2.2])
print(c)
print(c.type())
d = torch.FloatTensor(2)
print(d)
e = torch.IntTensor([2.2])
print(e)

data = np.ones(3)
print(data)
f = torch.from_numpy(data)    # 将numpy转换成tensor
print(f)

# Dim=2,Linear input/batch
g = torch.randn(2, 3)    # 随机正太分布
print(g)
print(g.shape)
print(g.size())
print(g.size(0))
print(g.size(1))
print(g.shape[1])


# Dim=3 RNN input/Batch
h = torch.rand(3, 2, 3)    # 随机均匀分布
print(h)
print(h.shape)
print(h[0])
print(h[1])

print(list(h.shape))
with open(r'E:\我的图片\南昌工程学院毕业照片\照片\1.jpg', mode='rb') as f:
    img = f.read()
    print(img)

lena = mpimg.imread(r'E:\我的图片\南昌工程学院毕业照片\照片\1.jpg')
print(lena)
print(type(lena))
ch = torch.from_numpy(lena)
print(ch)
print(ch.shape)
print(lena.shape)
plt.imshow(lena)
im = Image.open(r'E:\我的图片\南昌工程学院毕业照片\照片\1.jpg')
im.show()

# Dim=4 CNN:[b,c,h,w]
# 下面解释为2张照片，每张照片通道数为3，长宽为28×28
i = torch.rand(2, 3, 28, 28)    # 照片数 通道数(彩色图片为3) 图片长 图片宽

print(i)



# Mixed
j = torch.rand(2, 3, 28, 28)
print(j.numel())    # number of enement 2*3*28*28
print(j.dim())

