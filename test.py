import torch
import models

x = torch.rand(2, 3)-0.5
x.requires_grad_(True)

m = models.NoisyReLU(a=.9, b=1.1, inplace=True)
z = x.cuda()
m.cuda()

print('input: ', z)

res = m(z)
print('output: ', res)

o = res.sum()

o.backward()
print('input.grad: ', x.grad)