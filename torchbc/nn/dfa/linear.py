import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import linear
from einops import repeat

from torchbc.nn import DirectFeedbackModule
from torchbc.autograd.dfa import LinearGrad
from torchbc.nn.parameter import FeedbackParameter


class Linear(DirectFeedbackModule, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        feedback_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.feedback_input = None
        self.feedback_features = feedback_features

        self.feedback_weight = FeedbackParameter(
            data=torch.zeros(size=(in_features, feedback_features), requires_grad=False)
        )
        nn.init.kaiming_uniform_(self.feedback_weight)
        self.register_full_backward_hook(self.feedback_hook)

    def forward(self, x):
        return LinearGrad.apply(x, self.weight, self.bias)

    @staticmethod
    def feedback_hook(module, grad_input, grad_output):
        if grad_input[0] is None:
            return grad_input

        with torch.no_grad():
            grad_dfa = linear(
                input=module.feedback_input, weight=module.feedback_weight, bias=None
            )

        # Transformer case
        if len(grad_input[0].shape) > 2:
            repeat_times = grad_input[0].shape[-2]
            grad_dfa = repeat(grad_dfa, "b d -> b p d", p=repeat_times)

        return (grad_dfa,)

    def weight_init(self, method) -> None:
        if method == "xavier-uniform":
            nn.init.xavier_uniform_(self.weight)
        elif method == "xavier-normal":
            nn.init.xavier_normal_(self.weight)
        elif method == "kaiming-uniform":
            nn.init.kaiming_uniform_(self.weight)
        elif method == "kaiming-normal":
            nn.init.kaiming_normal_(self.weight)
        elif method == "zeros":
            nn.init.zeros_(self.weight)
        else:
            raise NotImplementedError(
                f"{method} is not defined in weight initialization."
            )
        nn.init.zeros_(self.bias)

    def backward_weight_init(
        self,
        method,
        sign_softmax_kappa: float = 1.0,
        sparse_dfa_sparse: int = 0,
        sparse_dfa_rank: int = 0,
    ) -> None:
        if method == "xavier-uniform":
            nn.init.xavier_uniform_(self.backward.weight)
        elif method == "xavier-normal":
            nn.init.xavier_normal_(self.backward.weight)
        elif method == "kaiming-uniform":
            nn.init.kaiming_uniform_(self.backward.weight)
        elif method == "kaiming-normal":
            nn.init.kaiming_normal_(self.backward.weight)
        elif method == "sign-softmax":
            weights = []
            for _ in range(self.backward.weight.shape[0]):
                b = torch.randn(1, self.backward.weight.shape[1]) * 1
                smb = (
                    torch.sign(b)
                    * torch.exp(sign_softmax_kappa * torch.abs(b))
                    / torch.sum(torch.exp(sign_softmax_kappa * torch.abs(b)))
                )
                weights.append(smb)
            self.backward.weight.data = torch.cat(weights)
        elif method == "sparse-dfa":
            input_size, output_size = self.backward.weight.shape

            if sparse_dfa_sparse:
                sqrt_fan_out = np.sqrt(
                    1.0 * output_size / np.sqrt(input_size * sparse_dfa_sparse)
                )
            else:
                sqrt_fan_out = np.sqrt(output_size)

            high = 1.0 / sqrt_fan_out
            low = -high

            fb = np.zeros(shape=self.backward.weight.shape)
            fb = np.transpose(fb)

            choices = range(input_size)
            counts = np.zeros(input_size)
            total_connects = 1.0 * sparse_dfa_sparse * sparse_dfa_rank
            connects_per = 1.0 * sparse_dfa_sparse * sparse_dfa_rank / input_size

            idxs = []

            if sparse_dfa_sparse and sparse_dfa_rank:
                assert sparse_dfa_sparse * sparse_dfa_rank <= input_size

                # pick rank sets of sparse indexes
                for ii in range(sparse_dfa_rank):
                    remaining_connects = total_connects - np.sum(counts)
                    pdf = (connects_per - counts) / remaining_connects
                    pdf = np.clip(pdf, 1e-6, 1.0)
                    pdf = pdf / np.sum(pdf)

                    choice = np.random.choice(
                        choices, sparse_dfa_sparse, replace=False, p=pdf
                    )
                    counts[choice] += 1.0
                    idxs.append(choice)

                # create our masks
                masks = []
                for ii in range(sparse_dfa_rank):
                    masks.append(np.zeros(shape=(output_size, input_size)))

                for ii in range(output_size):
                    choice = np.random.choice(range(len(idxs)))
                    idx = idxs[choice]
                    masks[choice][ii][idx] = 1.0

                # multiply mask by random rank 1 matrix.
                for ii in range(sparse_dfa_rank):
                    tmp1 = np.random.uniform(low, high, size=(output_size, 1))
                    tmp2 = np.random.uniform(low, high, size=(1, input_size))
                    fb = fb + masks[ii] * np.dot(tmp1, tmp2)

                # rank fix
                fb = fb * (high / np.std(fb))
                fb = fb.T

            elif sparse_dfa_sparse:
                mask = np.zeros(shape=(output_size, input_size))
                for ii in range(output_size):
                    idx = np.random.choice(
                        choices, size=sparse_dfa_sparse, replace=False
                    )
                    mask[ii][idx] = 1.0

                mask = mask.T
                fb = np.random.uniform(low, high, size=(input_size, output_size))
                fb = fb * mask

            elif sparse_dfa_rank:
                fb = np.zeros(shape=(input_size, output_size))
                for ii in range(sparse_dfa_rank):
                    tmp1 = np.random.uniform(low, high, size=(input_size, 1))
                    tmp2 = np.random.uniform(low, high, size=(1, output_size))
                    fb = fb + np.dot(tmp1, tmp2)
                # rank fix
                fb = fb * (high / np.std(fb))

            else:
                fb = np.random.uniform(low, high, size=(input_size, output_size))
            self.backward.weight.data = torch.tensor(
                fb, dtype=type(self.backward.weight)
            )
        else:
            raise NotImplementedError(
                f"{method} is not defined in backward weight initialization."
            )


class BottleneckLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super(BottleneckLinear, self).__init__(in_features, out_features, bias)
        self.loss_gradient = None
        self.backward = nn.Linear(
            in_features=out_features, out_features=in_features, bias=False
        )
        self.backward.weight.requires_grad = False

        # Default weight initialization
        # self.weight_init('zeros')
        self.backward_weight_init("xavier-uniform")
        self.register_full_backward_hook(self.dfa_backward_hook)

    def weight_init(self, method) -> None:
        if method == "xavier-uniform":
            nn.init.xavier_uniform_(self.weight)
        elif method == "xavier-normal":
            nn.init.xavier_normal_(self.weight)
        elif method == "kaiming-uniform":
            nn.init.kaiming_uniform_(self.weight)
        elif method == "kaiming-normal":
            nn.init.kaiming_normal_(self.weight)
        elif method == "zeros":
            nn.init.zeros_(self.weight)
        else:
            raise NotImplementedError(
                f"{method} is not defined in weight initialization."
            )
        nn.init.zeros_(self.bias)

    def backward_weight_init(
        self,
        method,
        sign_softmax_kappa: float = 1.0,
        sparse_dfa_sparse: int = 0,
        sparse_dfa_rank: int = 0,
    ) -> None:
        if method == "xavier-uniform":
            nn.init.xavier_uniform_(self.backward.weight)
        elif method == "xavier-normal":
            nn.init.xavier_normal_(self.backward.weight)
        elif method == "kaiming-uniform":
            nn.init.kaiming_uniform_(self.backward.weight)
        elif method == "kaiming-normal":
            nn.init.kaiming_normal_(self.backward.weight)
        elif method == "sign-softmax":
            weights = []
            for _ in range(self.backward.weight.shape[0]):
                b = torch.randn(1, self.backward.weight.shape[1]) * 1
                smb = (
                    torch.sign(b)
                    * torch.exp(sign_softmax_kappa * torch.abs(b))
                    / torch.sum(torch.exp(sign_softmax_kappa * torch.abs(b)))
                )
                weights.append(smb)
            self.backward.weight.data = torch.cat(weights)
        elif method == "sparse-dfa":
            input_size, output_size = self.backward.weight.shape

            if sparse_dfa_sparse:
                sqrt_fan_out = np.sqrt(
                    1.0 * output_size / np.sqrt(input_size * sparse_dfa_sparse)
                )
            else:
                sqrt_fan_out = np.sqrt(output_size)

            high = 1.0 / sqrt_fan_out
            low = -high

            fb = np.zeros(shape=self.backward.weight.shape)
            fb = np.transpose(fb)

            choices = range(input_size)
            counts = np.zeros(input_size)
            total_connects = 1.0 * sparse_dfa_sparse * sparse_dfa_rank
            connects_per = 1.0 * sparse_dfa_sparse * sparse_dfa_rank / input_size

            idxs = []

            if sparse_dfa_sparse and sparse_dfa_rank:
                assert sparse_dfa_sparse * sparse_dfa_rank <= input_size

                # pick rank sets of sparse indexes
                for ii in range(sparse_dfa_rank):
                    remaining_connects = total_connects - np.sum(counts)
                    pdf = (connects_per - counts) / remaining_connects
                    pdf = np.clip(pdf, 1e-6, 1.0)
                    pdf = pdf / np.sum(pdf)

                    choice = np.random.choice(
                        choices, sparse_dfa_sparse, replace=False, p=pdf
                    )
                    counts[choice] += 1.0
                    idxs.append(choice)

                # create our masks
                masks = []
                for ii in range(sparse_dfa_rank):
                    masks.append(np.zeros(shape=(output_size, input_size)))

                for ii in range(output_size):
                    choice = np.random.choice(range(len(idxs)))
                    idx = idxs[choice]
                    masks[choice][ii][idx] = 1.0

                # multiply mask by random rank 1 matrix.
                for ii in range(sparse_dfa_rank):
                    tmp1 = np.random.uniform(low, high, size=(output_size, 1))
                    tmp2 = np.random.uniform(low, high, size=(1, input_size))
                    fb = fb + masks[ii] * np.dot(tmp1, tmp2)

                # rank fix
                fb = fb * (high / np.std(fb))
                fb = fb.T

            elif sparse_dfa_sparse:
                mask = np.zeros(shape=(output_size, input_size))
                for ii in range(output_size):
                    idx = np.random.choice(
                        choices, size=sparse_dfa_sparse, replace=False
                    )
                    mask[ii][idx] = 1.0

                mask = mask.T
                fb = np.random.uniform(low, high, size=(input_size, output_size))
                fb = fb * mask

            elif sparse_dfa_rank:
                fb = np.zeros(shape=(input_size, output_size))
                for ii in range(sparse_dfa_rank):
                    tmp1 = np.random.uniform(low, high, size=(input_size, 1))
                    tmp2 = np.random.uniform(low, high, size=(1, output_size))
                    fb = fb + np.dot(tmp1, tmp2)
                # rank fix
                fb = fb * (high / np.std(fb))

            else:
                fb = np.random.uniform(low, high, size=(input_size, output_size))
            self.backward.weight.data = torch.tensor(
                fb, dtype=type(self.backward.weight)
            )
        else:
            raise NotImplementedError(
                f"{method} is not defined in backward weight initialization."
            )

    def forward(self, x):
        return LinearGrad.apply(x, self.weight, self.bias)

    @staticmethod
    def dfa_backward_hook(module, grad_input, grad_output):
        if grad_input[0] is None:
            return grad_input

        with torch.no_grad():
            grad_dfa = module.backward(module.loss_gradient)

        if len(grad_input[0].shape) > 2:
            repeat_times = grad_input[0].shape[-2]
            grad_dfa = repeat(grad_dfa, "b d -> b p d", p=repeat_times)

        return (grad_dfa,)
