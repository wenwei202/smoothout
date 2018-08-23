import torch
import torch.nn as nn
import torch.nn.functional as F
class NoisyReLU(nn.Module):
    r"""Applies the rectified linear unit function element-wise, but with a uniformly random slope
    :math:`\text{ReLU}(x)= \max(0, x)\*rand(a,b)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
        a: the min slope
        b: the max slope

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = NoisyReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, a, b, inplace=False):
        super(NoisyReLU, self).__init__()
        self.a = a
        self.b = b
        self.inplace = inplace
        self.register_buffer('device_flag', torch.zeros(1,1))
        self.register_buffer('noise', None)

    def forward(self, input):
	# in case the batch size decreases at the last iteration of an epch
	shape = list(input.size())
        if self._buffers['device_flag'].is_cuda:
            if self._buffers['noise'] is not None:
                self._buffers['noise'].uniform_().mul_(self.b-self.a).add_(self.a)
            else:
                self._buffers['noise'] = torch.cuda.FloatTensor(input.size()).uniform_().mul_(self.b-self.a).add_(self.a)
        else:
            if self._buffers['noise'] is not None:
                self._buffers['noise'].uniform_().mul_(self.b-self.a).add_(self.a)
            else:
                self._buffers['noise'] = torch.FloatTensor(input.size()).uniform_().mul_(self.b-self.a).add_(self.a)
	#print('noise:', self._buffers['noise'])
        return F.threshold(input * torch.autograd.Variable(self._buffers['noise'][0:shape[0]]), 0, 0, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'a={}, b={}{}'.format(
            self.a, self.b, inplace_str
        )
