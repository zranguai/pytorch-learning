import torch
"""
1. Mean Squared Error
2. Cross Entropy Loss
    1. binary
    2. multi-class
"""
"""
# 一：autograd.grad

# Mean Squared Error
# 这里注意MSE于2范数相比，2范数有开根号但是这里没有开根号

# 使用pytorch进行简单的求导
# 这里pred = w * x + b
from torch.nn import functional as F
x = torch.ones(1)
w = torch.full([1], 2)
mse = F.mse_loss(torch.ones(1), x*w)    # 第一个参数pred 第二个参数label
print(torch.autograd.grad(mse, [w]))    # 第一个参数loss  第二个参数w1, w2, w3
"""
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn?
# w函数在初始化的时候没有设置他需要导数信息，pytorch在建图的时候标注torch不需要求导信息
"""
# 改变如下：告诉pytorch w需要梯度信息
w.requires_grad_()
print(torch.autograd.grad(mse, [w]))
"""
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# 更新之后还是会报错，因为pytorch是一个动态图
# 这里更新了w但是图还没有更新
# 因为pytorch是做一步计算一次图
"""
# 必须经过计算图的过程重新更新一遍
mse = F.mse_loss(torch.ones(1), x*w)    # 动态图的建图
print(torch.autograd.grad(mse, [w]))    # (tensor([2.]),) 图重新更新后可以计算出结果
print(mse)

"""

# 二：loss.backward
from torch.nn import functional as F
x = torch.ones(1)
w = torch.full([1], 2)
mse = F.mse_loss(torch.ones(1), x*w)


# torch.autograd.grad(mse, [w])

# w.requires_grad_()    # 使w获取梯度
#
# mse = F.mse_loss(torch.ones(1), x*w)    # 再次计算获取动态图
# # torch.autograd.grad(mse, [w])    # 1. 自动计算  再次计算梯度
#
# mse.backward()    # 2. 手动计算tensor([2.])
# print(w.grad)


"""
Gradient API
    1. 手动求导torch.autograd.grad(loss, [w1, w2, ...])    
        [w1 grad, w2 grad...]
    
    2. 自动求导loss.backward()    # 他返回的梯度信息不会返回而是附在每个梯度信息上面
        w1.grad
        w2.grad
"""

















