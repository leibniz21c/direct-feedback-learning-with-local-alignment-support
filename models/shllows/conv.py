import torch.nn as nn
import torchbc.nn as bcnn


class SHALLOW_CONV_CIFAR10(bcnn.Module):
    def __init__(self):
        super(SHALLOW_CONV_CIFAR10, self).__init__()
        # global flow
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc_relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=2048, out_features=10)

        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        x = self.fc_relu2(x)
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
    

class SHALLOW_CONV_CIFAR100(bcnn.Module):
    def __init__(self):
        super(SHALLOW_CONV_CIFAR100, self).__init__()
        # global flow
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc_relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=2048, out_features=100)

        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        x = self.fc_relu2(x)
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