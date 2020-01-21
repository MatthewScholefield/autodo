import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

from autodo.custom_models import BasicBlock
from autodo.model.attention_module import AttentionModule_stage3

IMG_WIDTH = 448
IMG_HEIGHT = 128


class BackBone(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        self.inplanes = 64  # 64
        super(BackBone, self).__init__()
        #        self.simplemodel = reallysimplemodel()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 64
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)  # 128
        self.attention_module3 = AttentionModule_stage3(256, 256, (16, 42))
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)  # 256
        self.attention_module4 = AttentionModule_stage3(512, 512, (8, 21))
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)  # 512

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
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 224x224
        x = self.bn0(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112
        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        x = self.attention_module3(x)
        x = self.layer3(x)  # 14x14
        x = self.attention_module4(x)
        x = self.layer4(x)  # 7x7
        return x


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch_functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                       diffY // 2, diffY - diffY // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


class MyUNet(nn.Module):
    """Mixture of previous classes"""

    def __init__(self, n_classes, device):
        super(MyUNet, self).__init__()
        self.device = device
        self.base_model = BackBone(BasicBlock, [3, 4, 6, 3]).to(device)

        self.conv0 = double_conv(6, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        self.up1 = up(1024 + 1024 + 2, 512)
        self.up2 = up(512 + 512, 256)
        self.up3 = up(256 + 128, 128)
        self.up4 = up(128 + 64, 64)
        self.up5 = up(64 + 6, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def get_mesh(self, batch_size, shape_x, shape_y):
        mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
        mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
        mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
        mesh = torch.cat([torch.tensor(mg_x).to(self.device), torch.tensor(mg_y).to(self.device)], 1)
        return mesh

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = self.get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :3, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(self.device)
        feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        mesh2 = self.get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x0)
        x = self.outc(x)
        return x
