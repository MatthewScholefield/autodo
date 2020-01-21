import torch.nn as nn

from autodo.model.attention_module import AttentionModule_stage3


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        self.inplanes = 64  # 64
        super(ResNet, self).__init__()
        #        self.simplemodel = reallysimplemodel()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 128
        self.attention_module3 = AttentionModule_stage3(128, 128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 256
        self.attention_module4 = AttentionModule_stage3(256, 256, (7, 7))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 512
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(8192 * block.expansion, 200)
        self.fc1 = nn.Linear(200, 6)
        # self.bn1 = nn.BatchNorm1d(300)
        #         self.fc2 = nn.Linear(300, 500)
        #         self.bn2 = nn.BatchNorm1d(500)
        #         self.fc3 = nn.Linear(500, 300)
        #         self.bn3 = nn.BatchNorm1d(300)
        #         self.fc4 = nn.Linear(300, 200)
        #         self.bn4 = nn.BatchNorm1d(200)
        #         self.fc5 = nn.Linear(200, 200)
        #         self.bn5 = nn.BatchNorm1d(200)
        # self.fc6 = nn.Linear(300, 3)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        ## to compensate for the meshgrid for next round
        # self.inplanes += 2
        return nn.Sequential(*layers)

    def forward(self, x):
        # batch_size = len(box_specs)
        # mesh = get_mesh(batch_size, x.shape[2], x.shape[3], box_specs)
        # x = torch.cat([x, mesh], dim = 1)

        x = self.conv1(x)  # 224x224
        x = self.bn0(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        # mesh = get_mesh(batch_size, x.shape[2], x.shape[3], box_specs)
        # x = torch.cat([x, mesh], 1)

        x = self.layer1(x)  # 56x56
        # mesh = get_mesh(batch_size, x.shape[2], x.shape[3], box_specs)
        # x = torch.cat([x, mesh], 1)

        x = self.layer2(x)  # 28x28
        x = self.attention_module3(x)

        # mesh = get_mesh(batch_size, x.shape[2], x.shape[3], box_specs)
        # x = torch.cat([x, mesh], 1)

        x = self.layer3(x)  # 14x14
        x = self.attention_module4(x)
        # mesh = get_mesh(batch_size, x.shape[2], x.shape[3], box_specs)
        # x = torch.cat([x, mesh], 1)

        x = self.layer4(x)  # 7x7
        # mesh = get_mesh(batch_size, x.shape[2], x.shape[3], box_specs)
        # x = torch.cat([x, mesh], 1)
        # x = self.avgpool(x)  # 1x1
        x = x.view(x.size(0), -1)
        X = self.relu(self.fc(x))
        X = self.fc1(X)
        return X
