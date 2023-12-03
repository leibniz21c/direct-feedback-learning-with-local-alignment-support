from typing import Union

import torch
import torch.nn as nn

from torchbc.nn import DirectFeedbackModule
from torchbc.autograd.dfa import Conv2dGrad
from torch.nn.common_types import _size_2_t
from torchbc.nn.parameter import FeedbackParameter


class Conv2d(DirectFeedbackModule, nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feedback_features: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None, 
        dtype=None
    ):
        super(Conv2d, self).__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            bias, 
            padding_mode, 
            device, 
            dtype
        )
        self.feedback_input = None
        self.feedback_features = feedback_features
        self.feedback_weight = FeedbackParameter(
            data=torch.zeros(
                size=(feedback_features, self.in_channels, self.kernel_size[0], self.kernel_size[1]), 
                requires_grad=False
            ),
            requires_grad=False,
        )
        nn.init.kaiming_uniform_(self.feedback_weight)
        self.register_full_backward_hook(self.feedback_hook)


    def forward(self, x):
        return Conv2dGrad.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @staticmethod
    def feedback_hook(module, grad_input, grad_output):
        if grad_input[0] is None:
            return grad_input
        else:
            with torch.no_grad():
                repeated_grad_output = module.feedback_input.unsqueeze(2).repeat(
                    1, 1, grad_output[0].shape[2]
                )
                repeated_grad_output = repeated_grad_output.unsqueeze(3).repeat(1, 1, 1, grad_output[0].shape[3])
                grad_dfa = torch.nn.grad.conv2d_input(
                    input_size=grad_input[0].shape,
                    weight=module.feedback_weight,
                    grad_output=repeated_grad_output,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                )

            return (grad_dfa,)
