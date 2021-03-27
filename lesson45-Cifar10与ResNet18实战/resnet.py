import torch
from torch import nn
from torch.nn import functional as F    # 这里F和nn经常是交叉使用的


class ResBlk(nn.Module):
    """
    resnet block：这里是resnet的一个基本模块
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        # we add stride support for resbok, which is distinct from tutorials.
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()    # nn.Sequential()本来是空的
        if ch_out != ch_in:    # 如果不相等就把他的ch_in变成ch_out, 也就是说:他这个是resnet的旁边短接线
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )
    # -------------------------------#
    # 疑问: python在实例化的时候为啥不用调用forward函数？
    # 因为pytorch在nn.modules中使用了__call__,里面实现了forward方法
    # 只要实例化对象就会自动调用__call__,当自己又没有__call__方法，所以调用父类方法，由于子类重写了forward方法
    # 所以优先调用子类的forward方法
    # -------------------------------#
    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))    # 这里经过卷积层，BN层， 然后经过relu层
        out = self.bn2(self.conv2(out))    # 这里经过卷积层，BN层

        # short cut.    # 这里是短接
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        out = self.extra(x) + out    # element-wise add:
        out = F.relu(out)    # 最后再经过relu层输出
        print('这里打印下out看看', out.shape)
        return out




class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]    # 注意这里h,w是变化的
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)    # 这里视频是self.blk4 = ResBlk(512, 1024)

        self.outlayer = nn.Linear(512*1*1, 10)    # 最后再跟一个全连接层

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))    # 先经过一个卷积层，后面再跟一个relu函数, 经过后x.shape = [128, 64, 10, 10]
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)    # 经过这层后x.shape = torch.Size([128, 128, 5, 5])
        x = self.blk2(x)    # 经过这层后x.shape = torch.Size([128, 256, 3, 3])
        x = self.blk3(x)    # 经过这层后x.shape = torch.Size([128, 512, 2, 2])
        x = self.blk4(x)    # 经过这层后x.shape = torch.Size([128, 512, 2, 2])


        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)    # 经过这层后x.shape = torch.Size([128, 512])    x.size(0) = 128
        x = self.outlayer(x)    # 经过一个全连接层  经过这层后x.shape = torch.Size([128, 10])


        return x



def main():
    # ResBlk
    blk = ResBlk(64, 128, stride=4)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block:', out.shape)    # block: torch.Size([2, 128, 8, 8])

    # ResNet18
    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet:', out.shape)    # resnet: torch.Size([2, 10])




if __name__ == '__main__':
    main()



# ---------------ResNet18模型----------------------------#
"""
ResNet18(
  (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (blk1): ResBlk(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (extra): Sequential(
      (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (blk2): ResBlk(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (extra): Sequential(
      (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (blk3): ResBlk(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (extra): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (blk4): ResBlk(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (extra): Sequential()
  )
  (outlayer): Linear(in_features=512, out_features=10, bias=True)
)
"""
# ----------------------------------------------------------------------#
