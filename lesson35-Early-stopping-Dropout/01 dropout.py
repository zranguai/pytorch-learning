import torch


net_droped = torch.nn.Sequential(
    torch.nn.Linear(784, 200),
    torch.nn.Dropout(0.5),    # drop 50% of the neuron    (在两层之间断掉一些层)
    torch.nn.ReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.Dropout(0.5),    # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(200, 10),
)

"""
在训练是需要加上Dropout()

但是在test/val是不需要Dropout()
例如:
for epoch in range(epochs):
    # train
    net_dropped.train()
    for batch_idx, (data, targt) in enumerate(train_loader):
    ...
    
    net_dropped.eval()    # 在测试是需要加上这句话去掉dropout
    test_loss = 0
    correct = 0
    for data, target in test_loader:
    
"""















