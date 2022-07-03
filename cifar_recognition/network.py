from torch import nn
from torch import torch

class VGG(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5, w: int = 32, h: int = 32) -> None:
        super().__init__()
        a = int(w/32)
        b = int(h/32)

        # 定义构建网络的操作：conv、relu、pool、fc(linear)、dropout...

        # 1
        self.conv2d_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu_1 = nn.ReLU(True)

        # 2
        self.conv2d_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_2 = nn.ReLU(True)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu_3 = nn.ReLU(True)

        # 4
        self.conv2d_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu_4 = nn.ReLU(True)

        #
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 5
        self.conv2d_5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu_5 = nn.ReLU(True)

        # 6
        self.conv2d_6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_6 = nn.ReLU(True)

        # 7
        self.conv2d_7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_7 = nn.ReLU(True)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 8
        self.conv2d_8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu_8 = nn.ReLU(True)

        # 9
        self.conv2d_9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_9 = nn.ReLU(True)

        # 10
        self.conv2d_10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_10 = nn.ReLU(True)

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 11
        self.conv2d_11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_11 = nn.ReLU(True)

        # 12
        self.conv2d_12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_12 = nn.ReLU(True)

        # 13
        self.conv2d_13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_13 = nn.ReLU(True)

        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 14
        self.fc_1 = nn.Linear(512 * a * b, 4096)
        self.relu_14 = nn.ReLU(True)

        self.dropout_1 = nn.Dropout(p=dropout)
        
        # 15
        self.fc_2 = nn.Linear(4096, 4096)
        self.relu_15 = nn.ReLU(True)

        self.dropout_2 = nn.Dropout(p=dropout)

        # 16
        self.fc_out = nn.Linear(4096, num_classes)

        # 对网络的权重层进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_1(x)    # 1. n x 3 x 32a x 32b  ---→ n x 64 x 32a x 32b
        x = self.relu_1(x)

        x = self.conv2d_2(x)    # 2. n x 64 x 32a x 32b ---→ n x 64 x 32a x 32b
        x = self.relu_2(x)

        x = self.pool_1(x)      #    n x 64 x 32a x 32b ---→ n x 64 x 16a x 16b

        x = self.conv2d_3(x)    # 3. n x 64 x 16a x 16b ---→ n x 128 x 16a x 16b
        x = self.relu_3(x)

        x = self.conv2d_4(x)    # 4. n x 128 x 16a x 16b ---→ n x 128 x 16a x 16b
        x = self.relu_4(x)

        x = self.pool_2(x)

        x = self.conv2d_5(x)    # 5
        x = self.relu_5(x)

        x = self.conv2d_6(x)    # 6
        x = self.relu_6(x)

        x = self.conv2d_7(x)    # 7
        x = self.relu_7(x)

        x = self.pool_3(x)

        x = self.conv2d_8(x)    # 8
        x = self.relu_8(x)

        x = self.conv2d_9(x)    # 9
        x = self.relu_9(x)

        x = self.conv2d_10(x)   # 10
        x = self.relu_10(x)

        x = self.pool_4(x)

        x = self.conv2d_11(x)   # 11
        x = self.relu_11(x)

        x = self.conv2d_12(x)   # 12
        x = self.relu_12(x)

        x = self.conv2d_13(x)   # 13
        x = self.relu_13(x)

        x = self.pool_5(x)

        x = torch.flatten(x, 1)

        x = self.fc_1(x)        # 14
        x = self.relu_14(x)

        #x = self.dropout_1(x)

        x = self.fc_2(x)        # 15
        x = self.relu_15(x)

        #x = self.dropout_2(x)

        x = self.fc_out(x)      # 16

        return x

if __name__ == '__main__':
    net = VGG(w=32, h=32)
    x = torch.rand(1, 3, 32, 32)
    x = net(x)
    print(x.detach().numpy())