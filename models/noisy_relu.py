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
        # a flag
        self.register_buffer('device_flag', torch.zeros(1,1))

    def forward(self, input):
        if self._buffers['device_flag'].is_cuda:
            noise = torch.cuda.FloatTensor(input.size()).uniform_() * (self.b-self.a) + self.a
        else:
            noise = torch.FloatTensor(input.size()).uniform_() * (self.b-self.a) + self.a
        return F.threshold(input * noise, 0, 0, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'a={}, b={}{}'.format(
            self.a, self.b, inplace_str
        )
