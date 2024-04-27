import torch

rand_data = torch.rand(32, 20, 32)
rand_len = torch.randint(10, 20, (1, 32)).view(-1)
# rand_len
print(rand_len)
print(rand_len.size())
print(range(rand_len[range(32)]))
print(rand_data[range(32), rand_len, :].size())
print(rand_data[range(32), range(rand_len[:]), :])
print(rand_data[range(32), range(rand_len), :].size())
print()
