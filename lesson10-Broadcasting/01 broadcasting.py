# broadcasting     ？？？？
import torch
"""
expand
without copying data

insert 1 dim ahead
expand dims with size 1 to same size
feature maps:[4, 32, 14, 14]
bias:[32, 1, 1] => [1, 32, 1, 1] => [4, 32, 14, 14]    bias的扩张
"""
# situation 1
# [4, 32, 14, 14]
# [1, 32, 1, 1] => [4, 32, 14, 14]

# situation2
# [4, 32, 14, 14]
# [14, 14] => [1, 1, 14, 14] => [4, 32, 14, 14]    # 可以先unsqueeze再expand

# situation3（不符合）
# [4, 32, 14, 14]
# [2, 32, 14, 14]

# a = torch.tensor([2, 32, 14, 14])
# # print(a)
# # print(a[:])

# a = torch.IntTensor(4, 3)
# b = torch.IntTensor(3)
# print(a)
# print(b)
"""
match from last dim
1. no dim
2. dim of size 1
"""










