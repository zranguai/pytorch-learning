import torch
a = torch.full([4], 1/4)
print(a)
print(a*torch.log2(a))
print(-(a*torch.log2(a)).sum())    # tensor(2.) 熵越高代表越稳定，没有惊喜度
b = torch.tensor([0.1, 0.1, 0.1, 0.7])
print(-(b*torch.log2(b)).sum())    # tensor(1.3568) higher uncertainty  惊喜度较高
c = torch.tensor([0.001, 0.001, 0.001, 0.999])
print(-(c*torch.log2(c)).sum())    # tensor(0.0313)  极度不稳定



















