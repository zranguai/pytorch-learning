import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import torch
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

# 画图
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)    # 将x这个图片和y这个图片拼接到一起
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)    # 把x, y的坐标送入Z函数里面得到z的坐标
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# 找最小值--初始点不同找的也不同
# [1., 0.], [-4, 0.], [4, 0.]
x = torch.tensor([4., 0.], requires_grad=True)    # 在这里不同的初始化权重更新的速率和最后得到的结果都不太同。所以说梯度下降法的初始化很关键
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):

    pred = himmelblau(x)    # x送进来得到预测值，目的是min这个预测值

    optimizer.zero_grad()    # 将梯度信息进行清零
    pred.backward()    # 生成x.grad和y.grad即：x和y的梯度信息
    optimizer.step()    # 将x,y的梯度进行更新

    if step % 2000 == 0:
        print('step {}: x = {}, f(x) = {}'
               .format(step, x.tolist(), pred.item()))
