import torch
from torch.nn.functional import conv2d
from torch.autograd import Function
from torch.nn.grad import conv2d_input, conv2d_weight


# Regular Backpropagation Algorithm
class Conv2dGrad(Function):
    @staticmethod
    def forward(context, input, weight, bias, stride, padding, dilation, groups):
        context.stride, context.padding, context.dilation, context.groups = (
            stride,
            padding,
            dilation,
            groups,
        )
        context.save_for_backward(input, weight, bias)
        return conv2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    @staticmethod
    def backward(context, grad_output):
        input, weight, bias = context.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Gradient input
        if context.needs_input_grad[0]:
            grad_input = conv2d_input(
                input_size=input.shape,
                weight=weight,
                grad_output=grad_output,
                stride=context.stride,
                padding=context.padding,
                dilation=context.dilation,
                groups=context.groups,
            )
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = conv2d_weight(
                input=input,
                weight_size=weight.shape,
                grad_output=grad_output,
                stride=context.stride,
                padding=context.padding,
                dilation=context.dilation,
                groups=context.groups,
            )
        # Gradient bias
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).sum(2).sum(1)

        # We return the same number of parameters
        return grad_input, grad_weight, grad_bias, None, None, None, None
