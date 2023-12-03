import torch
import torch.nn as nn

import torchbc.nn as bcnn
from torchbc.nn import dfa



class LAS_FC_MNIST(bcnn.Module):
    def __init__(self):
        super(LAS_FC_MNIST, self).__init__()
        # global flow
        self.fc1 = nn.Linear(1*28*28, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = dfa.Linear(1024, 1024, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = dfa.Linear(1024, 10, 10)

        # local flow
        self.lfc1 = nn.Linear(1024, 10, bias=False)

        self.lfc1.weight.data = torch.matmul(self.fc3.weight.data, self.fc2.weight.data)


    def forward(self, x, get_local_output=False):
        if get_local_output:
            with torch.no_grad():
                x = self.fc1(x)
                x1 = self.relu1(x)
                x = self.fc2(x1)
                x = self.relu2(x)
                global_output = self.fc3(x)
            return self.lfc1(x1), global_output
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x


    def fit(self, data, label, criteria, optimizer):
        # local flow update
        optimizer.zero_grad()
        local_output, global_output = self.forward(data, get_local_output=True)
        local_loss = criteria[0](local_output, global_output)
        local_loss.backward()
        optimizer.step()

        # align
        self.fc2.feedback_weight.data = self.lfc1.weight.data.T
        self.fc3.feedback_weight.data = self.fc3.weight.data.T

        # global flow update
        optimizer.zero_grad()
        output = self.forward(data)
        global_loss = criteria[1](output, label)
        global_error = self.get_global_errors(global_loss, output)
        self.direct_feedback(global_error)
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": local_loss, 
            "global_loss": global_loss,
            "local_output": local_output,
            "output": output,
        }
    
    
class LAS_FC_CIFAR10(bcnn.Module):
    def __init__(self):
        super(LAS_FC_CIFAR10, self).__init__()
        # global flow
        self.fc1 = nn.Linear(3*32*32, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = dfa.Linear(4096, 4096, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = dfa.Linear(4096, 4096, 10)
        self.relu3 = nn.ReLU()
        self.fc4 = dfa.Linear(4096, 10, 10)

        # local flow
        self.lfc1 = nn.Linear(4096, 10, bias=False)
        self.lfc2 = nn.Linear(4096, 10, bias=False)

        self.lfc1.weight.data = torch.matmul(torch.matmul(self.fc4.weight.data, self.fc3.weight.data), self.fc2.weight.data)
        self.lfc2.weight.data = torch.matmul(self.fc4.weight.data, self.fc3.weight.data)


    def forward(self, x, get_local_output=False):
        if get_local_output:
            with torch.no_grad():
                x = self.fc1(x)
                x1 = self.relu1(x)
                x = self.fc2(x1)
                x2 = self.relu2(x)
                x = self.fc3(x2)
                x = self.relu3(x)
                global_output = self.fc4(x)
            return self.lfc1(x1), self.lfc2(x2), global_output
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x


    def fit(self, data, label, criteria, optimizer):
        # local flow update
        optimizer.zero_grad()
        local_output1, local_output2, global_output = self.forward(data, get_local_output=True)
        local_loss1 = criteria[0](local_output1, global_output)
        local_loss2 = criteria[0](local_output2, global_output)
        local_loss1.backward()
        local_loss2.backward()
        optimizer.step()

        # align
        self.fc2.feedback_weight.data = self.lfc1.weight.data.T
        self.fc3.feedback_weight.data = self.lfc2.weight.data.T
        self.fc4.feedback_weight.data = self.fc4.weight.data.T

        # global flow update
        optimizer.zero_grad()
        output = self.forward(data)
        global_loss = criteria[1](output, label)
        global_error = self.get_global_errors(global_loss, output)
        self.direct_feedback(global_error)
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": (local_loss1, local_loss2), 
            "global_loss": global_loss,
            "local_output": (local_output1, local_output2),
            "output": output,
        }


class LAS_FC_CIFAR100(bcnn.Module):
    def __init__(self):
        super(LAS_FC_CIFAR100, self).__init__()
        # global flow
        self.fc1 = nn.Linear(3*32*32, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = dfa.Linear(4096, 4096, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = dfa.Linear(4096, 4096, 100)
        self.relu3 = nn.ReLU()
        self.fc4 = dfa.Linear(4096, 100, 100)

        # local flow
        self.lfc1 = nn.Linear(4096, 100, bias=False)
        self.lfc2 = nn.Linear(4096, 100, bias=False)

        self.lfc1.weight.data = torch.matmul(torch.matmul(self.fc4.weight.data, self.fc3.weight.data), self.fc2.weight.data)
        self.lfc2.weight.data = torch.matmul(self.fc4.weight.data, self.fc3.weight.data)


    def forward(self, x, get_local_output=False):
        if get_local_output:
            with torch.no_grad():
                x = self.fc1(x)
                x1 = self.relu1(x)
                x = self.fc2(x1)
                x2 = self.relu2(x)
                x = self.fc3(x2)
                x = self.relu3(x)
                global_output = self.fc4(x)
            return self.lfc1(x1), self.lfc2(x2), global_output
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x


    def fit(self, data, label, criteria, optimizer):
        # local flow update
        optimizer.zero_grad()
        local_output1, local_output2, global_output = self.forward(data, get_local_output=True)
        local_loss1 = criteria[0](local_output1, global_output)
        local_loss2 = criteria[0](local_output2, global_output)
        local_loss1.backward()
        local_loss2.backward()
        optimizer.step()

        # align
        self.fc2.feedback_weight.data = self.lfc1.weight.data.T
        self.fc3.feedback_weight.data = self.lfc2.weight.data.T
        self.fc4.feedback_weight.data = self.fc4.weight.data.T

        # global flow update
        optimizer.zero_grad()
        output = self.forward(data)
        global_loss = criteria[1](output, label)
        global_error = self.get_global_errors(global_loss, output)
        self.direct_feedback(global_error)
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": (local_loss1, local_loss2), 
            "global_loss": global_loss,
            "local_output": (local_output1, local_output2),
            "output": output,
        }