import math
from abc import ABCMeta, abstractmethod, abstractstaticmethod
import torch
import torch.nn as nn
from torch.autograd import grad

from torchbc.nn.parameter import FeedbackParameter


class SequentialFeedbackModule(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x):
        pass

    def forward_update(self):
        raise NotImplementedError("Must implement forward_update method.")

    def feedback_update(self):
        raise NotImplementedError("Must implement feedback_update method.")

    @torch.no_grad()
    def get_weight_alignment(self):
        x1, x2 = torch.reshape(self.weight.data, (-1, )), torch.reshape(self.feedback_weight.data.T, (-1, ))
        x1, x2 = x1 / torch.norm(x1), x2 / torch.norm(x2)
        return (180. / math.pi) * torch.arccos(torch.clip(torch.dot(x1, x2), -1., 1.))


class DirectFeedbackModule(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractstaticmethod
    def feedback_hook(module, grad_input, grad_output):
        pass

    def forward_update(self):
        raise NotImplementedError("Must implement forward_update method.")

    def feedback_update(self):
        raise NotImplementedError("Must implement feedback_update method.")


class Module(nn.Module):
    def __init__(self) -> None:
        super(Module, self).__init__()

        # Direct feedback propagation
        self._direct_feedback_prop = False
        self._reverse_ordered_layers = None
        self._reverse_ordered_weights = None

        # Direct feedback weight alignment
        self._ordered_layers = None
        self._ordered_weights = None

    def forward(self, x):
        raise NotImplementedError("Must implement it.")

    ##### decoupling parameters #####
    def forward_parameters(self):
        params = []
        for param in super(Module, self).parameters():
            if not isinstance(param, FeedbackParameter):
                params.append(param)
        return params

    def feedback_parameters(self):
        params = []
        for param in super(Module, self).parameters():
            if isinstance(param, FeedbackParameter):
                params.append(param)
        return params

    ##### Sequential feedback family #####
    def seq_forward_update(self):
        for module in self.modules():
            if isinstance(module, SequentialFeedbackModule):
                module.forward_update()

    def seq_feedback_update(self, **kwargs):
        for module in self.modules():
            if isinstance(module, SequentialFeedbackModule):
                module.feedback_update(**kwargs)

    def seq_update(self):
        for module in self.modules():
            if isinstance(module, SequentialFeedbackModule):
                module.forward_update()
                module.feedback_update()

    def seq_forward_decay(self):
        for module in self.modules():
            if isinstance(module, SequentialFeedbackModule) and hasattr(
                module, "weight_decay"
            ):
                module.weight.data -= module.weight_decay * module.weight.data

    def seq_feedback_decay(self):
        for module in self.modules():
            if isinstance(module, SequentialFeedbackModule) and hasattr(
                module, "weight_decay"
            ):
                module.feedback_weight.data -= (
                    module.weight_decay * module.feedback_weight.data
                )

    def seq_decay(self):
        for module in self.modules():
            if isinstance(module, SequentialFeedbackModule) and hasattr(
                module, "weight_decay"
            ):
                module.weight.data -= module.weight_decay * module.weight.data
                module.feedback_weight.data -= (
                    module.weight_decay * module.feedback_weight.data
                )

    ##### Direct feedback family #####
    # Direct feedback alignment
    def get_global_errors(self, loss, outputs):
        errors = grad(loss, outputs, retain_graph=True)[0]
        return errors

    def direct_feedback(self, x):
        for module in self.modules():  # For all descendent modules
            if isinstance(module, DirectFeedbackModule):
                module.feedback_input = x

    def dir_forward_update(self):
        for module in self.modules():
            if isinstance(module, DirectFeedbackModule):
                module.forward_update()

    def dir_feedback_update(self):
        for module in self.modules():
            if isinstance(module, DirectFeedbackModule):
                module.feedback_update()

    def dir_update(self):
        for module in self.modules():
            if isinstance(module, DirectFeedbackModule):
                module.forward_update()
                module.feedback_update()

    def set_fully_connected_direct_feedback_weight_alignment(self, ordered_layers):
        self._ordered_layers = ordered_layers
        self._ordered_weights = []

        for layer in self._ordered_layers:
            self._ordered_weights.append(layer.weight)

    def get_dir_weight_alignments(self):
        alignments = []

        for i in range(len(self._ordered_layers) - 1):
            x1 = torch.reshape(self._ordered_layers[i].weight.data, (-1, ))
            x2 = torch.reshape(torch.mm(self._ordered_layers[i + 1].feedback_weight.data, self._ordered_layers[i].feedback_weight.data.T), (-1, ))
            x1, x2 = x1 / torch.norm(x1), x2 / torch.norm(x2)

            alignment = (180. / math.pi) * torch.arccos(torch.clip(torch.dot(x1, x2), -1., 1.))
            alignments.append(alignment.item())

        x1 = torch.reshape(self._ordered_layers[-1].weight.data, (-1, ))
        x2 = torch.reshape(self._ordered_layers[-1].feedback_weight.data.T, (-1, ))
        x1, x2 = x1 / torch.norm(x1), x2 / torch.norm(x2)
        alignment = (180. / math.pi) * torch.arccos(torch.clip(torch.dot(x1, x2), -1., 1.))
        alignments.append(alignment.item())

        return alignments
    
    def get_weight_alignments(self):
        alignments = []

        for module in self.modules():
            if isinstance(module, SequentialFeedbackModule):
                alignments.append(module.get_weight_alignment().item())

        return alignments
    

    # Direct feedback propagation
    # [DEPRECATED]
    def update_direct_propagated_feedback_weight(self):
        if not self._direct_feedback_prop:
            raise ValueError("Module is not direct feedback propagation mode.")

        weight_product = self._reverse_ordered_weights[0].data.T
        self._reverse_ordered_layers[0].feedback_weight.data = weight_product
        for i in range(1, len(self._reverse_ordered_weights)):
            weight_product = torch.matmul(
                self._reverse_ordered_weights[i].data.T, weight_product
            )
            self._reverse_ordered_layers[i].feedback_weight.data = weight_product

    def set_fully_connected_direct_feedback_propagation(self, ordered_layers):
        self._direct_feedback_prop = True
        self._reverse_ordered_layers = tuple(reversed(ordered_layers))
        self._reverse_ordered_weights = []

        for layer in self._reverse_ordered_layers:
            self._reverse_ordered_weights.append(layer.weight)
