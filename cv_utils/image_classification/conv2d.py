import torch
import torch.nn as nn

from functools import reduce
from operator import __add__

# Source https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))
    
    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)