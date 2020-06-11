import torch
import torch.nn as nn
from torchsummary import summary


class MnistModel:
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


class MnistModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=14, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=14 * 4 * 4, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        # x is 1x28x28
        x = torch.relu(self.conv1(x))  # 10x24x24
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 10x12x12

        x = torch.relu(self.conv2(x))  # 14x8x8
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 14x4x4

        x = x.view(-1, 14 * 4 * 4)
        x = torch.relu(self.fc1(x))  # 500

        x = self.fc2(x)  # 10
        return torch.log_softmax(x, dim=1)


class CnnKoch2015(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 10, 1, )
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 7, 1)
        self.conv2_bn = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 4, 1)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 4, 1)
        self.conv4_bn = nn.BatchNorm2d(256)

        self._init_convolution_layers([self.conv1, self.conv2, self.conv3, self.conv4])

        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.fc_final = nn.Linear(in_features=4096, out_features=1)
        self._init_linear_layers([self.fc1, self.fc_final])

    def _init_convolution_layers(self, conv_layers):
        for c in conv_layers:
            nn.init.normal_(c.weight, mean=0, std=0.01)
            nn.init.normal_(c.bias, mean=0.5, std=0.01)

    def _init_linear_layers(self, linear_layers):
        for l in linear_layers:
            nn.init.normal_(l.weight, mean=0, std=0.2)
            nn.init.normal_(l.bias, mean=0.5, std=0.01)

    def forward(self, x1, x2):
        left = self._cnn_forward(x1)
        right = self._cnn_forward(x2)

        x = torch.abs(left - right)
        x = self.fc_final(x)

        return x

    def _cnn_forward(self, x):
        # x is 1x105x105
        x = self.conv1(x)  # 64x96x96
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 64x48x48

        x = self.conv2(x)  # 128x42x42
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 128x21x21

        x = self.conv3(x)  # 128x18x18
        x = self.conv3_bn(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 128x9x9

        x = self.conv4(x)  # 256x6x6 = 9216
        x = self.conv4_bn(x)
        x = torch.relu(x)
        x = x.view(-1, 256 * 6 * 6)

        x = self.fc1(x)  # 4096
        x = torch.sigmoid(x)
        return x

    def summary(self, device="cpu"):
        summary(self, input_size=[(1, 105, 105), (1, 105, 105)], device=device)


if __name__ == '__main__':
    net = CnnKoch2015()
    for name, param in net.named_parameters():
        print(name)
    net.summary()
