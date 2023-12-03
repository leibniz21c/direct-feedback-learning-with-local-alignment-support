import torch.nn as nn

import torchbc.nn as bcnn


class SHALLOW_FC_MNIST(bcnn.Module):
    def __init__(self):
        super(SHALLOW_FC_MNIST, self).__init__()
        # global flow
        self.fc1 = nn.Linear(1*28*28, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 10)

        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False


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
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": None, 
            "global_loss": global_loss,
            "alignments": None,
            "local_output": None,
            "output": output,
        }
    

class SHALLOW_FC_CIFAR10(bcnn.Module):
    def __init__(self):
        super(SHALLOW_FC_CIFAR10, self).__init__()
        # global flow
        self.fc1 = nn.Linear(3*32*32, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 4096)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(4096, 10)

        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False



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
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": None, 
            "global_loss": global_loss,
            "alignments": None,
            "local_output": None,
            "output": output,
        }


class SHALLOW_FC_CIFAR100(bcnn.Module):
    def __init__(self):
        super(SHALLOW_FC_CIFAR100, self).__init__()
        # global flow
        self.fc1 = nn.Linear(3*32*32, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 4096)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(4096, 100)

        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False


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
        global_loss.backward()
        optimizer.step()

        return {
            "local_loss": None, 
            "global_loss": global_loss,
            "alignments": None,
            "local_output": None,
            "output": output,
        }