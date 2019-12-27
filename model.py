import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class G_net(nn.Module):
    def __init__(self, inchannel=3, outchannel=3, ngf=64, reslayer=9):
        super(G_net, self).__init__()
        self.inconv = InConv(inchannel, ngf)
        self.down1 = DownSample(ngf, ngf * 2)
        self.down2 = DownSample(ngf * 2, ngf * 4)
        self.resnet = nn.Sequential()
        for i in range(reslayer):
            self.resnet.add_module('res{0}'.format(i), BasicBlock(ngf * 4, ngf * 4))
        self.up1 = UpSample(ngf * 4, ngf * 2)
        self.up2 = UpSample(ngf * 2, ngf)
        self.outconv = OutConv(ngf, outchannel)

    def forward(self, x):
        x = self.inconv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.resnet(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.outconv(x)
        return x


class InConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(InConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(inchannel, outchannel, kernel_size=7),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(inchannel, outchannel, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class UpSample(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up(x)


class D_net(nn.Module):
    def __init__(self, inchannel=3, ndf=64, layer=4):
        super(D_net, self).__init__()
        self.conv = nn.Sequential()
        for i in range(layer):
            self.conv.add_module('conv{0}'.format(i), nn.Conv2d(inchannel, ndf, kernel_size=3, stride=2, padding=1))
            self.conv.add_module('norm{0}'.format(i), nn.BatchNorm2d(ndf))
            self.conv.add_module('act{0}'.format(i), nn.LeakyReLU(0.2))
            inchannel = ndf
            ndf = ndf * 2
        self.conv.add_module('out', nn.Conv2d(inchannel, 1, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('act', nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class PatchLoss(nn.Module):
    def __init__(self):
        super(PatchLoss, self).__init__()

    def get_target_tensor(self, x, flag):
        flag = 1.0 if flag is True else 0.0
        tensor = torch.tensor(flag)
        return tensor.expand_as(x).to(device)

    def forward(self, x, flag):
        tensor = self.get_target_tensor(x, flag)
        return nn.BCELoss()(x, tensor)


if __name__ == '__main__':
    gnet = G_net(reslayer=2)
    dnet = D_net()
    criterian = PatchLoss()
    input = torch.randn([1, 3, 119, 119])
    print(input.shape)
    output = gnet(input)
    print(output.shape)

    output = dnet(input)
    print(output.shape)
    print(output)
    loss = criterian(output, False)
    print(loss)

