import torch
from torch.utils.data import DataLoader    # DataLoader是为了能够批量加载数据
from torchvision import datasets    # 从torchvision中导入数据集
from torchvision import transforms
from torch import nn, optim

from lenet5 import Lenet5
from resnet import ResNet18


def main():
    batchsz = 128    # 这里是batch-size

    # torchvision中提供一些已有的数据集 #  第一个参数:自定目录，第二个参数:Train=True, transform：对数据做些变换
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=False)    # download=True:可以自动的download
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)    # Dataloader：方便一次加载多个. shuffle:加载的时候随机换一下

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=False)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)


    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)    # x: torch.Size([128, 3, 32, 32]) label: torch.Size([128])

    device = torch.device('cuda')    # 后面可以使用GPU计算
    # model = Lenet5().to(device)
    model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)    # loss函数他包含softmax, 因为是分类任务所以采用crossentropy
    optimizer = optim.Adam(model.parameters(), lr=1e-3)    # 优化器把网络里的参数传给他
    print(model)

    for epoch in range(1000):

        model.train()    # 模型为train模式
        for batchidx, (x, label) in enumerate(cifar_train):    # 从每个epoch里的batch_size
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)    # 转换到cuda上面来


            logits = model(x)    # 他与predict的区别是是否经过softmax操作
            # logits: [b, 10]
            # label:  [b]    # label不需要probality
            # loss: tensor scalar    # 长度为0的标量
            loss = criteon(logits, label)    # 这个label是y

            # backprop
            optimizer.zero_grad()    # 如果不清0就是累加的效果
            loss.backward()
            optimizer.step()    # 更新weight,更新的weight写进optimizer里面


        print(epoch, 'loss:', loss.item())    # 对于标量,使用item()把他转换成Numpy

        # test
        model.eval()    # 模型为test模式
        with torch.no_grad():    # 这一步是告诉不需要构建梯度(不需要构建图)
            # test
            total_correct = 0    # 正确的数量
            total_num = 0    # 总的数量
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)



if __name__ == '__main__':
    main()
