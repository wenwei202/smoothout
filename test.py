import torch
import models
import numpy as np

m = models.NoisyReLU(a=.9, b=1.1, inplace=True)

for i in range(3):
	if 2==i:
		x = np.random.rand(2,2).astype('f') - 0.5
	else:
		x = np.random.rand(3,2).astype('f') - 0.5
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

print('=========== eval ============')

m.eval()
for i in range(2):
	if 1==i:
		x = np.random.rand(5,2).astype('f') - 0.5
	else:
		x = np.random.rand(1,2).astype('f') - 0.5
	x = torch.from_numpy(x)

	z = x.cuda()
	z = torch.autograd.Variable(z)
	m.cuda()

	print('input: ', z)

	res = m(z)
	print('output: ', res)
	print('-------------------')