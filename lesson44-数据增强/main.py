import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from visdom import Visdom

# Data argumentation
# ---------------------------------------#
# 这些操作在torchvision包里面
# 1. Flip：翻转
# 2. Rotate
# 3. Random Move & Crop
# 4. GAN : 生成更多的样本
# 5. Noise: N(0, 0.001)加高斯白噪声
# ---------------------------------------#

batch_size=200
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([    # Compose的操作类似于nn.Sequential里面
                       transforms.RandomHorizontalFlip(),    # 水平角度的翻转    （随机翻转-可能翻转也有可能不翻转）
                       transforms.RandomVerticalFlip(),    # 垂直方向
                       transforms.RandomRotation(15),    # 旋转方向，参数为旋转的度数
                       transforms.RandomRotation([90, 180, 270]),    # 随机的从90度180度270度中挑一个角度旋转
                       transforms.Resize([32, 32]),    # 传入的参数为list
                       transforms.RandomCrop([28, 28]),    # 裁剪
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),    # x 转换成x'
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)



class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                   legend=['loss', 'acc.']))
global_step = 0

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        global_step += 1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    viz.line([[test_loss, correct / len(test_loader.dataset)]],
             [global_step], win='test', update='append')
    viz.images(data.view(-1, 1, 28, 28), win='x')
    viz.text(str(pred.detach().cpu().numpy()), win='pred',
             opts=dict(title='pred'))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
