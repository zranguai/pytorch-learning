import torch
from torch import nn
from torch import optim

# -----------------------------------#
# 使用nn.Module的好处
# 1. 所有的常用的方法都在里面,比如: Linear/Relu/Sigmoid等
# 2. 使用nn.Sequential()容器[sequential是串行的意思], 不管是nn.Module中的还是自己写的都可以在这里使用
# 3. nn.Module可以自动管理parameters
# 4. modules: all nodes / children: direct children
# 5. to(device) （第84行）
# 6. save and load(第90行)
# 7. train/test的方便的切换(第87行)
# 8. implement own layer 实现自己的类(第31 / 第41 行)    只有class才能写到nn.Sequential里面去[第48行]
# -----------------------------------#

class MyLinear(nn.Module):

    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()

        # requires_grad = True
        self.w = nn.Parameter(torch.randn(outp, inp))    # nn.Parameter会自动地将torch.tensor通过nn.Parameter加到nn.parameter()里面去
        self.b = nn.Parameter(torch.randn(outp))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x


class Flatten(nn.Module):    # 将所有的打平

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)    # -1表示将其他所有的打平



class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(1, 16, stride=1, padding=1),
                                 nn.MaxPool2d(2, 2),
                                 Flatten(),    # 实现自己的类，里面只能写类
                                 nn.Linear(1*14*14, 10))

    def forward(self, x):
        return self.net(x)


class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()

        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x)



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #  使用nn.Sequential()容器[sequential是串行的意思], 不管是nn.Module中的还是自己写的都可以在这里使用
        self.net = nn.Sequential(BasicNet(),
                                 nn.ReLU(),
                                 nn.Linear(3, 2))

    def forward(self, x):
        return self.net(x)





def main():
    device = torch.device('cuda')
    net = Net()
    net.to(device)    # .to()会返回net引用(和原来的net引用一样)    --->但是对于tensor操作来说不是这样的

    # train
    net.train()
    # test
    net.eval()

    # net.load_state_dict(torch.load('ckpt.mdl'))    # 在开始的时候要加载模型
    #
    #
    # torch.save(net.state_dict(), 'ckpt.mdl')    # 在模型断电或者中断保存模型的当前状态

    for name, t in net.named_parameters():
        print('parameters:', name, t.shape)    # 打印里面地parameters:权重和bias

    for name, m in net.named_children():    # 打印net Sequential的类
        print('children:', name, m)


    for name, m in net.named_modules():
        print('modules:', name, m)



if __name__ == '__main__':
    main()