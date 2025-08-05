import torch
from torch import nn

class WCPNet(nn.Module):
    def __init__(self, out_target=4):
        super(WCPNet, self).__init__()
        # First stage
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(16)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(32)
        self.leaky_relu2 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second stage
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.InstanceNorm2d(64)
        self.leaky_relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.InstanceNorm2d(64)
        self.leaky_relu4 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third stage
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(32)
        self.tanh = nn.Tanh()

        # Fourth stage
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.InstanceNorm2d(32)
        self.leaky_relu6 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.norm7 = nn.InstanceNorm2d(32)
        self.leaky_relu7 = nn.LeakyReLU()

        # First stage
        self.conv_transpose1_v = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.norm1_v = nn.InstanceNorm2d(16)
        self.leaky_relu1_v = nn.LeakyReLU()

        # Second stage
        self.conv1_v = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm2_v = nn.InstanceNorm2d(16)
        self.leaky_relu2_v = nn.LeakyReLU()
        self.conv2_v = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm3_v = nn.InstanceNorm2d(16)
        self.leaky_relu3_v = nn.LeakyReLU()

        # third stage
        self.conv_transpose3_v = nn.Conv2d(48, 4, kernel_size=4, stride=2, padding=1)
        self.norm4_v = nn.InstanceNorm2d(4)
        self.leaky_relu4_v = nn.LeakyReLU()

        #fin
        self.conv_fin = nn.Conv2d(4, out_target, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # First stage
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.leaky_relu2(x)
        x = self.pool1(x)
        skip = x
        
        # Second stage
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.leaky_relu4(x)
        x = self.pool2(x)

        # Third stage
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.tanh(x)

        # Fourth stage
        x = self.conv6(x)
        x = self.norm6(x)
        x = self.leaky_relu6(x)
        x = self.conv7(x)
        x = self.norm7(x)
        x = self.leaky_relu7(x)

        # First stage
        x = self.conv_transpose1_v(x)
        x = self.norm1_v(x)
        x = self.leaky_relu1_v(x)

        # Second stage
        x = self.conv1_v(x)
        x = self.norm2_v(x)
        x = self.leaky_relu2_v(x)
        x = self.conv2_v(x)
        x = self.norm3_v(x)
        x = self.leaky_relu3_v(x)

        #cat
        x = torch.cat((x, skip), dim=1)

        # third
        x = self.conv_transpose3_v(x)
        x = self.norm4_v(x)
        x = self.leaky_relu4_v(x)

        return x


def main():
    encoder = WCPNet(out_target=4)
    input = torch.randn(1, 7, 512, 512)
    output = encoder(input)
    print(output.shape)


if __name__ == '__main__':
    main()