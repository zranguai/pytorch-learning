import torch
from torch import nn    # 任何类都要继承自nn.module
from torch.nn import functional as F

# -------------------------#
# 卷积神经网络最简单的版本
# -------------------------#

class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(    # 将网络结构写在这个nn.Sequential中
            # x: [b, 3, 32, 32] => [b, 16, ]    # b是batch_size
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),    # in_ch, out_ch,    得到的是[b, 16, 28, 28]
            # pool层不改变chanel的数量,只改变长宽
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),    # 得到的是[b, 16, 14, 14]
            #
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),    # 得到的是[b, 16, 10, 10]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),    # 得到的是[b, 16, 5, 5]
            #
        )

        # flatten: 打平要单独写,因为在nn.Sequential中没有打平的代码
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(32*5*5, 32),    # 这里是全连接层, 这里的2*16*5*5是从前面的卷积层下来的
            nn.ReLU(),    # 加激活函数
            # nn.Linear(120, 84),
            # nn.ReLU(),
            nn.Linear(32, 10)    # nn. 这些类都是小写的, F.这些类都是大写的
            ## nn.的类需要初始化(在forward就不需要写这些信息), F.的类不需要初始化(因为他是一个算子)
        )


#------- 这部分的代码时因为作者想要测试输出的维度是多少-----#
        # [b, 3, 32, 32]
        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        # [b, 16, 5, 5]
        print('conv out:', out.shape)

        # # use Cross Entropy Loss
        # self.criteon = nn.CrossEntropyLoss()    # 对于逼近或者regression使用这个
        # self.criteon = nn.CrossEntropyLoss()    # 对于分类问题用这个
#------------------------------------------------------------#


    def forward(self, x):    # 只要往前面走一下, pytorch会自动计算前向的路径,backward的时候就不用自己实现他会自己计算
        """

        :param x: [b, 3, 32, 32]
        :return:
        """
        batchsz = x.size(0)    # 这里是batch-size         x.size(0)==x.shape(0), 根据main()函数,这里batchsz=2
        # [b, 3, 32, 32] => [b, 16, 5, 5]    # 经过卷积层后
        x = self.conv_unit(x)    # 这样写时就已经调用了实例的forward方法
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, 32*5*5)    # 这里打平是为了送进全连接层,  这里的32*5*5也可以写成-1
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)    # 这里送进全连接层    这里叫logits是因为一般他要送进softmax,softmax前的变量一般叫做logits(sigmod之前的函数也叫logits)


        # # [b, 10], 在10维上做softmax
        # pred = F.softmax(logits, dim=1)    # 因为在softmax中的前面的criteon中会自动有softmax,所以这一步可以不用写???

        # loss = self.criteon(logits, y)

        return logits

def main():

    net = Lenet5()

    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet out:', out.shape)




if __name__ == '__main__':
    main()