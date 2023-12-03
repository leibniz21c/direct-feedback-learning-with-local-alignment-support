import torch
from torch.nn.functional import linear
from torch.autograd import Function


# Regular Backpropagation algorithm
class LinearGrad(Function):
    @staticmethod
    def forward(context, input, weight, bias=None):
        context.save_for_backward(input, weight, bias)
        return linear(input, weight, bias)

    @staticmethod
    def backward(context, grad_output):
        input, weight, bias = context.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Gradient input
        if context.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
        # Gradient weights
        if context.needs_input_grad[1]:
            if len(grad_output.shape) < 2:
                grad_weight = torch.outer(grad_output, input)
            else:
                grad_weight = torch.mm(
                    grad_output.flatten(end_dim=-2).T, input.flatten(end_dim=-2)
                )

        # Gradient bias
        if bias is not None and context.needs_input_grad[2]:
            if len(grad_output.shape) < 2:
                grad_bias = grad_output
            else:
                grad_bias = grad_output.flatten(end_dim=-2).sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
