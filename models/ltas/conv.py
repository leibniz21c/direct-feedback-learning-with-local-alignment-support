import torch
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

        # local flow
        self.lconv1 = nn.Conv2d(in_channels=96, out_channels=num_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.lgp1 = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.lconv2 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.lgp2 = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.lfc1 = nn.Linear(in_features=4096, out_features=num_classes, bias=False)
        self.lfc2 = nn.Linear(in_features=2048, out_features=num_classes, bias=False)


    def forward(self, x, get_local_output=False):
        if get_local_output:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.relu1(x)
                cx1 = self.maxpool1(x)

                x = self.conv2(cx1)
                x = self.relu2(x)
                cx2 = self.maxpool2(x)

                x = self.conv3(cx2)
                x = self.relu3(x)
                x = self.maxpool3(x)
                x1 = self.flatten(x)

                x = self.fc1(x1)
                x2 = self.fc_relu1(x)
                
            return (
                self.lgp1(self.lconv1(cx1)).squeeze(dim=-1).squeeze(dim=-1),
                self.lgp2(self.lconv2(cx2)).squeeze(dim=-1).squeeze(dim=-1),
                self.lfc1(x1),
                self.lfc2(x2),
            )
            

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
        local_output1, local_output2, local_output3, local_output4 = self.forward(data, get_local_output=True)
        local_loss1, local_loss2, local_loss3, local_loss4 = (
            criteria[0](local_output1, label),
            criteria[0](local_output2, label),
            criteria[0](local_output3, label),
            criteria[0](local_output4, label),
        )
        local_loss1.backward()
        local_loss2.backward()
        local_loss3.backward()
        local_loss4.backward()
        optimizer.step()

        # align
        self.conv2.feedback_weight.data = self.lconv1.weight.data
        self.conv3.feedback_weight.data = self.lconv2.weight.data
        self.fc1.feedback_weight.data = self.lfc1.weight.data.T
        self.fc2.feedback_weight.data = self.lfc2.weight.data.T
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
            "local_loss": (local_loss1, local_loss2, local_loss3, local_loss4), 
            "global_loss": global_loss,
            "local_output": (local_output1, local_output2, local_output3, local_output4),
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

        # local flow
        self.lconv1 = nn.Conv2d(in_channels=96, out_channels=num_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.lgp1 = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.lconv2 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.lgp2 = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.lfc1 = nn.Linear(in_features=4096, out_features=num_classes, bias=False)
        self.lfc2 = nn.Linear(in_features=2048, out_features=num_classes, bias=False)


    def forward(self, x, get_local_output=False):
        if get_local_output:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                cx1 = self.maxpool1(x)

                x = self.conv2(cx1)
                x = self.bn2(x)
                x = self.relu2(x)
                cx2 = self.maxpool2(x)

                x = self.conv3(cx2)
                x = self.bn3(x)
                x = self.relu3(x)
                x = self.maxpool3(x)
                x1 = self.flatten(x)

                x = self.fc1(x1)
                x2 = self.fc_relu1(x)
                
            return (
                self.lgp1(self.lconv1(cx1)).squeeze(dim=-1).squeeze(dim=-1),
                self.lgp2(self.lconv2(cx2)).squeeze(dim=-1).squeeze(dim=-1),
                self.lfc1(x1),
                self.lfc2(x2),
            )
            

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
        local_output1, local_output2, local_output3, local_output4 = self.forward(data, get_local_output=True)
        local_loss1, local_loss2, local_loss3, local_loss4 = (
            criteria[0](local_output1, label),
            criteria[0](local_output2, label),
            criteria[0](local_output3, label),
            criteria[0](local_output4, label),
        )
        local_loss1.backward()
        local_loss2.backward()
        local_loss3.backward()
        local_loss4.backward()
        optimizer.step()

        # align
        self.conv2.feedback_weight.data = self.lconv1.weight.data
        self.conv3.feedback_weight.data = self.lconv2.weight.data
        self.fc1.feedback_weight.data = self.lfc1.weight.data.T
        self.fc2.feedback_weight.data = self.lfc2.weight.data.T
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
            "local_loss": (local_loss1, local_loss2, local_loss3, local_loss4), 
            "global_loss": global_loss,
            "local_output": (local_output1, local_output2, local_output3, local_output4),
            "output": output,
        }
    

class LTAS_CONV_CIFAR10(Conv):
    def __init__(self):
        super(LTAS_CONV_CIFAR10, self).__init__(num_classes=10)


class LTAS_CONV_CIFAR100(Conv):
    def __init__(self):
        super(LTAS_CONV_CIFAR100, self).__init__(num_classes=100)
        

class LTAS_CONVBN_CIFAR10(ConvBN):
    def __init__(self):
        super(LTAS_CONVBN_CIFAR10, self).__init__(num_classes=10)


class LTAS_CONVBN_CIFAR100(ConvBN):
    def __init__(self):
        super(LTAS_CONVBN_CIFAR100, self).__init__(num_classes=100)