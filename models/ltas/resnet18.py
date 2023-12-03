import torch
import torch.nn as nn
from torchbc import nn as bcnn
from torchbc.nn import dfa


class ResNet(bcnn.Module):
    def __init__(self, num_classes : int = 1000) -> None:
        super(ResNet, self).__init__()
        
        # input block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # layer1
        self.layer1_block1_conv1 = dfa.Conv2d(64, 64, feedback_features=num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block1_bn1 = nn.BatchNorm2d(64)
        self.layer1_block1_relu1 = nn.ReLU()
        self.layer1_block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block1_bn2 = nn.BatchNorm2d(64)
        self.layer1_block1_relu2 = nn.ReLU()

        self.layer1_block2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block2_bn1 = nn.BatchNorm2d(64)
        self.layer1_block2_relu1 = nn.ReLU()
        self.layer1_block2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block2_bn2 = nn.BatchNorm2d(64)
        self.layer1_block2_relu2 = nn.ReLU()

        # layer2
        self.layer2_block1_conv1 = dfa.Conv2d(64, 128, feedback_features=num_classes, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_block1_bn1 = nn.BatchNorm2d(128)
        self.layer2_block1_relu1 = nn.ReLU()
        self.layer2_block1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block1_bn2 = nn.BatchNorm2d(128)
        self.layer2_block1_downsample_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.layer2_block1_downsample_bn1 = nn.BatchNorm2d(128)
        self.layer2_block1_relu2 = nn.ReLU()

        self.layer2_block2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block2_bn1 = nn.BatchNorm2d(128)
        self.layer2_block2_relu1 = nn.ReLU()
        self.layer2_block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block2_bn2 = nn.BatchNorm2d(128)
        self.layer2_block2_relu2 = nn.ReLU()

        # layer3
        self.layer3_block1_conv1 = dfa.Conv2d(128, 256, feedback_features=num_classes, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_block1_bn1 = nn.BatchNorm2d(256)
        self.layer3_block1_relu1 = nn.ReLU()
        self.layer3_block1_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block1_bn2 = nn.BatchNorm2d(256)
        self.layer3_block1_downsample_conv1 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.layer3_block1_downsample_bn1 = nn.BatchNorm2d(256)
        self.layer3_block1_relu2 = nn.ReLU()

        self.layer3_block2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block2_bn1 = nn.BatchNorm2d(256)
        self.layer3_block2_relu1 = nn.ReLU()
        self.layer3_block2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block2_bn2 = nn.BatchNorm2d(256)
        self.layer3_block2_relu2 = nn.ReLU()

        # layer4
        self.layer4_block1_conv1 = dfa.Conv2d(256, 512, feedback_features=num_classes, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_block1_bn1 = nn.BatchNorm2d(512)
        self.layer4_block1_relu1 = nn.ReLU()
        self.layer4_block1_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block1_bn2 = nn.BatchNorm2d(512)
        self.layer4_block1_downsample_conv1 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.layer4_block1_downsample_bn1 = nn.BatchNorm2d(512)
        self.layer4_block1_relu2 = nn.ReLU()

        self.layer4_block2_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block2_bn1 = nn.BatchNorm2d(512)
        self.layer4_block2_relu1 = nn.ReLU()
        self.layer4_block2_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block2_bn2 = nn.BatchNorm2d(512)
        self.layer4_block2_relu2 = nn.ReLU(inplace=False)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = dfa.Linear(in_features=512, out_features=num_classes, feedback_features=num_classes, bias=True)

        # local flow
        self.local_layer1_block1_conv1 = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.local_layer2_block1_conv1 = nn.Conv2d(64, num_classes, kernel_size=3, stride=2, padding=1, bias=False)
        self.local_layer3_block1_conv1 = nn.Conv2d(128, num_classes, kernel_size=3, stride=2, padding=1, bias=False)
        self.local_layer4_block1_conv1 = nn.Conv2d(256, num_classes, kernel_size=3, stride=2, padding=1, bias=False)

        self.local_layer1_block1_conv1_gp = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.local_layer2_block1_conv1_gp = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.local_layer3_block1_conv1_gp = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.local_layer4_block1_conv1_gp = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, x, get_local_output=False):
        if get_local_output:
            with torch.no_grad():
                # input layer
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)

                # layer 1
                local_input_layer1_block1_conv1 = identity = x
                x = self.layer1_block1_conv1(x)
                x = self.layer1_block1_bn1(x)
                x = self.layer1_block1_relu1(x)
                x = self.layer1_block1_conv2(x)
                x = self.layer1_block1_bn2(x)
                x += identity
                x = self.layer1_block1_relu2(x)

                identity = x
                x = self.layer1_block2_conv1(x)
                x = self.layer1_block2_bn1(x)
                x = self.layer1_block2_relu1(x)
                x = self.layer1_block2_conv2(x)
                x = self.layer1_block2_bn2(x)
                x += identity
                x = self.layer1_block2_relu2(x)

                # layer 2
                local_input_layer2_block1_conv1 = identity = x
                x = self.layer2_block1_conv1(x)
                x = self.layer2_block1_bn1(x)
                x = self.layer2_block1_relu1(x)
                x = self.layer2_block1_conv2(x)
                x = self.layer2_block1_bn2(x)
                x += self.layer2_block1_downsample_bn1(self.layer2_block1_downsample_conv1(identity))
                x = self.layer2_block1_relu2(x)

                identity = x
                x = self.layer2_block2_conv1(x)
                x = self.layer2_block2_bn1(x)
                x = self.layer2_block2_relu1(x)
                x = self.layer2_block2_conv2(x)
                x = self.layer2_block2_bn2(x)
                x += identity
                x = self.layer2_block2_relu2(x)

                # layer 3
                local_input_layer3_block1_conv1 = identity = x
                x = self.layer3_block1_conv1(x)
                x = self.layer3_block1_bn1(x)
                x = self.layer3_block1_relu1(x)
                x = self.layer3_block1_conv2(x)
                x = self.layer3_block1_bn2(x)
                x += self.layer3_block1_downsample_bn1(self.layer3_block1_downsample_conv1(identity))
                x = self.layer3_block1_relu2(x)

                identity = x
                x = self.layer3_block2_conv1(x)
                x = self.layer3_block2_bn1(x)
                x = self.layer3_block2_relu1(x)
                x = self.layer3_block2_conv2(x)
                x = self.layer3_block2_bn2(x)
                x += identity
                x = self.layer3_block2_relu2(x)

                # layer 4
                local_input_layer4_block1_conv1 = identity = x

            return (
                self.local_layer1_block1_conv1_gp(self.local_layer1_block1_conv1(local_input_layer1_block1_conv1)).squeeze(dim=-1).squeeze(dim=-1),
                self.local_layer2_block1_conv1_gp(self.local_layer2_block1_conv1(local_input_layer2_block1_conv1)).squeeze(dim=-1).squeeze(dim=-1),
                self.local_layer3_block1_conv1_gp(self.local_layer3_block1_conv1(local_input_layer3_block1_conv1)).squeeze(dim=-1).squeeze(dim=-1),
                self.local_layer4_block1_conv1_gp(self.local_layer4_block1_conv1(local_input_layer4_block1_conv1)).squeeze(dim=-1).squeeze(dim=-1),
            )
        
        # input layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # layer 1
        identity = x
        x = self.layer1_block1_conv1(x)
        x = self.layer1_block1_bn1(x)
        x = self.layer1_block1_relu1(x)
        x = self.layer1_block1_conv2(x)
        x = self.layer1_block1_bn2(x)
        x += identity
        x = self.layer1_block1_relu2(x)
        
        identity = x
        x = self.layer1_block2_conv1(x)
        x = self.layer1_block2_bn1(x)
        x = self.layer1_block2_relu1(x)
        x = self.layer1_block2_conv2(x)
        x = self.layer1_block2_bn2(x)
        x += identity
        x = self.layer1_block2_relu2(x)

        # layer 2
        identity = x
        x = self.layer2_block1_conv1(x)
        x = self.layer2_block1_bn1(x)
        x = self.layer2_block1_relu1(x)
        x = self.layer2_block1_conv2(x)
        x = self.layer2_block1_bn2(x)
        x += self.layer2_block1_downsample_bn1(self.layer2_block1_downsample_conv1(identity))
        x = self.layer2_block1_relu2(x)
        
        identity = x
        x = self.layer2_block2_conv1(x)
        x = self.layer2_block2_bn1(x)
        x = self.layer2_block2_relu1(x)
        x = self.layer2_block2_conv2(x)
        x = self.layer2_block2_bn2(x)
        x += identity
        x = self.layer2_block2_relu2(x)

        # layer 3
        identity = x
        x = self.layer3_block1_conv1(x)
        x = self.layer3_block1_bn1(x)
        x = self.layer3_block1_relu1(x)
        x = self.layer3_block1_conv2(x)
        x = self.layer3_block1_bn2(x)
        x += self.layer3_block1_downsample_bn1(self.layer3_block1_downsample_conv1(identity))
        x = self.layer3_block1_relu2(x)
        
        identity = x
        x = self.layer3_block2_conv1(x)
        x = self.layer3_block2_bn1(x)
        x = self.layer3_block2_relu1(x)
        x = self.layer3_block2_conv2(x)
        x = self.layer3_block2_bn2(x)
        x += identity
        x = self.layer3_block2_relu2(x)

        # layer 4
        identity = x
        x = self.layer4_block1_conv1(x)
        x = self.layer4_block1_bn1(x)
        x = self.layer4_block1_relu1(x)
        x = self.layer4_block1_conv2(x)
        x = self.layer4_block1_bn2(x)
        x += self.layer4_block1_downsample_bn1(self.layer4_block1_downsample_conv1(identity))
        x = self.layer4_block1_relu2(x)
        
        identity = x
        x = self.layer4_block2_conv1(x)
        x = self.layer4_block2_bn1(x)
        x = self.layer4_block2_relu1(x)
        x = self.layer4_block2_conv2(x)
        x = self.layer4_block2_bn2(x)
        x += identity
        x = self.layer4_block2_relu2(x)
        
        x = self.avgpool(x).squeeze(dim=-1).squeeze(dim=-1)
        x = self.fc(x)

        return x
    
    def fit(self, data, label, criteria, optimizer):
        # local flow update
        optimizer.zero_grad()
        (
            local_output_layer1_block1_conv1,
            local_output_layer2_block1_conv1,
            local_output_layer3_block1_conv1,
            local_output_layer4_block1_conv1,
        ) = self.forward(data, get_local_output=True)

        local_loss_layer1_block1_conv1 = criteria[0](local_output_layer1_block1_conv1, label); local_loss_layer1_block1_conv1.backward()
        local_loss_layer2_block1_conv1 = criteria[0](local_output_layer2_block1_conv1, label); local_loss_layer2_block1_conv1.backward()
        local_loss_layer3_block1_conv1 = criteria[0](local_output_layer3_block1_conv1, label); local_loss_layer3_block1_conv1.backward()
        local_loss_layer4_block1_conv1 = criteria[0](local_output_layer4_block1_conv1, label); local_loss_layer4_block1_conv1.backward()
        optimizer.step()

        # align
        self.layer1_block1_conv1.feedback_weight.data = self.local_layer1_block1_conv1.weight.data
        self.layer2_block1_conv1.feedback_weight.data = self.local_layer2_block1_conv1.weight.data
        self.layer3_block1_conv1.feedback_weight.data = self.local_layer3_block1_conv1.weight.data
        self.layer4_block1_conv1.feedback_weight.data = self.local_layer4_block1_conv1.weight.data
        self.fc.feedback_weight.data = self.fc.weight.data.T

        # global flow update
        optimizer.zero_grad()
        output = self.forward(data)
        global_loss = criteria[1](output, label)
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


class LTAS_RESNET18_CIFAR10(ResNet):
    def __init__(self):
        super(LTAS_RESNET18_CIFAR10, self).__init__(num_classes=10)


class LTAS_RESNET18_CIFAR100(ResNet):
    def __init__(self):
        super(LTAS_RESNET18_CIFAR100, self).__init__(num_classes=100)


class LTAS_RESNET18_TINYIMAGENET(ResNet):
    def __init__(self):
        super(LTAS_RESNET18_TINYIMAGENET, self).__init__(num_classes=200)