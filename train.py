import torch
from torch import nn
from torch.utils.data import DataLoader
from datagen import DatasetFromFolder
from os.path import join
import os
from model import G_net, D_net, PatchLoss, device
from utils import AverageMeter, visualize

root = 'data'

print_freq = 10
weight = 10
epochs = 1000
lr = 0.0001
batch_size = 32
input_channel = 3
output_channel = 3
ngf = 64
ndf = 64
g_layer = 9
d_layer = 3
check = 'best_checkpoint.tar'

train_set = DatasetFromFolder(root)
val_set = DatasetFromFolder(root)
train_loader = DataLoader(train_set, batch_size, True)
tval_loader = DataLoader(val_set, batch_size, True)

if os.path.exists(check):
    print('load checkpoint')
    checkpoint = torch.load(check)
    net_g = checkpoint[0]
    net_d = checkpoint[1]
else:
    print('train from init')
    net_g = G_net(input_channel, output_channel, ngf, g_layer).to(device)
    net_d = D_net(input_channel + output_channel, ndf, d_layer).to(device)

criterionGAN = PatchLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

optimzer_g = torch.optim.Adam(net_g.parameters(), lr)
optimzer_d = torch.optim.Adam(net_d.parameters(), lr)


def train():
    for epoch in range(epochs):
        avg_loss_g = AverageMeter()
        avg_loss_d = AverageMeter()
        min_loss_g = float('inf')
        min_loss_d = float('inf')
        for i, data in enumerate(train_loader):
            img_a, img_b = data[0].to(device), data[1].to(device)
            fake_b = net_g(img_a)

            # update discriminator
            optimzer_d.zero_grad()

            fake = torch.cat((img_a, fake_b), 1)
            out_fake = net_d(fake.detach())
            loss_fake = criterionGAN(out_fake, False)

            real = torch.cat((img_a, img_b), 1)
            out_real = net_d(real)
            loss_real = criterionGAN(out_real, True)

            loss_d = (loss_fake + loss_real) * 0.5
            loss_d.backward()
            optimzer_d.step()

            # update generator
            optimzer_g.zero_grad()

            fake = torch.cat((img_a, fake_b), 1)
            out_fake = net_d(fake)
            loss_g = criterionGAN(out_fake, True)

            loss_L1 = criterionL1(img_b, fake_b) * weight
            loss_g = loss_g + loss_L1
            loss_g.backward()
            optimzer_g.step()

            avg_loss_d.update(loss_d)
            avg_loss_g.update(loss_g)

            if i % print_freq == 0:
                print('epoch {}/{}'.format(epoch, i))
                print('loss: lossg {0} lossd{1} avgg{2} avgd{3}'.format(avg_loss_g.val, avg_loss_d.val, avg_loss_g.avg, avg_loss_d.avg))
                if loss_g < min_loss_g and loss_d < min_loss_d:
                    torch.save((net_g, net_d), check)
        vis()


def vis():
    visualize(net_g, train_loader)


if __name__ == '__main__':
    train()
