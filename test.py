import torch
import models
import numpy as np

m = models.NoisyReLU(a=.9, b=1.1, inplace=True)

for _ in range(1000):
	x = np.random.rand(30000,2000).astype('f') - 0.5
	x = torch.from_numpy(x)

	z = x.cuda()
	z = torch.autograd.Variable(z, requires_grad=True)
	m.cuda()

	print('input: ', z)

	res = m(z)
	print('output: ', res)

	o = res.sum()

	o.backward()
	print('input.grad: ', z.grad)
	print('-------------------')
