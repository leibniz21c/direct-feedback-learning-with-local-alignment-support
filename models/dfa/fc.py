import torch.nn as nn

import torchbc.nn as bcnn
from torchbc.nn import dfa


class DFA_FC_MNIST(bcnn.Module):
    def __init__(self):
        super(DFA_FC_MNIST, self).__init__()
        # global flow
        self.fc1 = nn.Linear(1*28*28, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = dfa.Linear(1024, 1024, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = dfa.Linear(1024, 10, 10)

        # set alignments
        self.set_fully_connected_direct_feedback_weight_alignment(
            ordered_layers=(self.fc2, self.fc3)
        )


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x


    def fit(self, data, label, criteria, optimizer):

        # global flow update
        optimizer.zero_grad()
        output = self.forward(data)
        global_loss = criteria(output, label)
        global_error = self.get_global_errors(global_loss, output)
        self.direct_feedback(global_error)
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": None, 
            "global_loss": global_loss,
            "local_output": None,
            "output": output,
        }


class DFA_FC_CIFAR10(bcnn.Module):
    def __init__(self):
        super(DFA_FC_CIFAR10, self).__init__()
        # global flow
        self.fc1 = nn.Linear(3*32*32, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = dfa.Linear(4096, 4096, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = dfa.Linear(4096, 4096, 10)
        self.relu3 = nn.ReLU()
        self.fc4 = dfa.Linear(4096, 10, 10)

        # set alignments
        self.set_fully_connected_direct_feedback_weight_alignment(
            ordered_layers=(self.fc2, self.fc3, self.fc4)
        )


    def forward(self, x, get_local_output=False):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x


    def fit(self, data, label, criteria, optimizer):

        # global flow update
        optimizer.zero_grad()
        output = self.forward(data)
        global_loss = criteria(output, label)
        global_error = self.get_global_errors(global_loss, output)
        self.direct_feedback(global_error)
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": None, 
            "global_loss": global_loss,
            "local_output": None,
            "output": output,
        }


class DFA_FC_CIFAR100(bcnn.Module):
    def __init__(self):
        super(DFA_FC_CIFAR100, self).__init__()
        # global flow
        self.fc1 = nn.Linear(3*32*32, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = dfa.Linear(4096, 4096, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = dfa.Linear(4096, 4096, 100)
        self.relu3 = nn.ReLU()
        self.fc4 = dfa.Linear(4096, 100, 100)

        # set alignments
        self.set_fully_connected_direct_feedback_weight_alignment(
            ordered_layers=(self.fc2, self.fc3, self.fc4)
        )


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x


    def fit(self, data, label, criteria, optimizer):

        # global flow update
        optimizer.zero_grad()
        output = self.forward(data)
        global_loss = criteria(output, label)
        global_error = self.get_global_errors(global_loss, output)
        self.direct_feedback(global_error)
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": None, 
            "global_loss": global_loss,
            "local_output": None,
            "output": output,
        }