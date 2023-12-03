import torch.nn as nn
import torchbc.nn as bcnn
from torchbc.nn import dfa


class Conv(bcnn.Module):
    def __init__(self, num_classes: int = 1000):
        super(Conv, self).__init__()
        # global flow
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = dfa.Conv2d(in_channels=96, out_channels=128, feedback_features=num_classes, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = dfa.Conv2d(in_channels=128, out_channels=256, feedback_features=num_classes, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc1 = dfa.Linear(in_features=4096, out_features=2048, feedback_features=num_classes)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = dfa.Linear(in_features=2048, out_features=2048, feedback_features=num_classes)
        self.fc_relu2 = nn.ReLU()
        self.fc3 = dfa.Linear(in_features=2048, out_features=num_classes, feedback_features=num_classes)


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


class ConvBN(bcnn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ConvBN, self).__init__()
        # global flow
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = dfa.Conv2d(in_channels=96, out_channels=128, feedback_features=num_classes, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = dfa.Conv2d(in_channels=128, out_channels=256, feedback_features=num_classes, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc1 = dfa.Linear(in_features=4096, out_features=2048, feedback_features=num_classes)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = dfa.Linear(in_features=2048, out_features=2048, feedback_features=num_classes)
        self.fc_relu2 = nn.ReLU()
        self.fc3 = dfa.Linear(in_features=2048, out_features=num_classes, feedback_features=num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
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
    

class DFA_CONV_CIFAR10(Conv):
    def __init__(self):
        super(DFA_CONV_CIFAR10, self).__init__(num_classes=10)


class DFA_CONV_CIFAR100(Conv):
    def __init__(self):
        super(DFA_CONV_CIFAR100, self).__init__(num_classes=100)


class DFA_CONVBN_CIFAR10(ConvBN):
    def __init__(self):
        super(DFA_CONVBN_CIFAR10, self).__init__(num_classes=10)


class DFA_CONVBN_CIFAR100(ConvBN):
    def __init__(self):
        super(DFA_CONVBN_CIFAR100, self).__init__(num_classes=100)