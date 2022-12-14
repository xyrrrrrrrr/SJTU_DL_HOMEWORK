import jittor as jt
from jittor import nn
from jittor import transform
import numpy as np

# use jittor to construct a CNN for cifar10
class mynet(nn.Module):
    def __init__(self, num_classes=10):
        super(mynet, self).__init__()
        self.conv1 = nn.Conv(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv(32, 64, 3, 1, 1)
        self.pool = nn.Pool(2, 2)
        self.fc1 = nn.Linear(8 * 8 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def execute(self, x):
        # x = self.pool(nn.relu(self.conv1(x)))
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.pool(x)
        x = self.pool(nn.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)

        self.extra = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride = stride),
            nn.BatchNorm2d(planes)
        )

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.extra(residual)
        out = self.relu(out)

        return out
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        self.blk1 = BasicBlock(64, 64, stride=2)
        self.blk2 = BasicBlock(64, 128, stride=2)
        self.blk3 = BasicBlock(128, 256, stride=2)
        self.blk4 = BasicBlock(256, 512, stride=2)
        self.relu = nn.ReLU()
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1,1))
        self.outlayer = nn.Linear(512*1*1, 10)

    def execute(self, x):

        x = self.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = self.AdaptiveAvgPool2d(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x

if __name__ == '__main__':
    x = jt.rand([1, 3, 32, 32])
    model = mynet()
    y = model(x)
    print(y)
