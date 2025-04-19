import sys

print("\n" + sys.version)

for num in range(0, 51):
    if num % 7 == 0:
        print(num)


import torch
torch.zeros(1).cuda()
print(torch.cuda.is_available())

A = torch.tensor([1, 2, 3])
print(A[:-1])
print(A[:-2])
print(A[-1])