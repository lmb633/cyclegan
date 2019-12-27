import torch
from torch import nn
from torch.utils.data import DataLoader
from datagen import DatasetFromFolder
import itertools
import os
from model import G_net, D_net, PatchLoss, device
from utils import AverageMeter, visualize, weights_init_normal

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
train_loader = DataLoader(train_set, batch_size, True)

if os.path.exists(check):
    print('load checkpoint')
    checkpoint = torch.load(check)
    netg_a2b = checkpoint[0]
    netg_b2a = checkpoint[1]
    netd_a = checkpoint[2]
    netd_b = checkpoint[3]
else:
    print('train from init')
    netg_a2b = G_net(input_channel, output_channel, ngf, g_layer).to(device)
    netg_b2a = G_net(input_channel, output_channel, ngf, g_layer).to(device)
    netd_a = D_net(input_channel + output_channel, ndf, d_layer).to(device)
    netd_b = D_net(input_channel + output_channel, ndf, d_layer).to(device)
    netg_a2b.apply(weights_init_normal)
    netg_b2a.apply(weights_init_normal)
    netd_a.apply(weights_init_normal)
    netd_b.apply(weights_init_normal)

criterionGAN = PatchLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

optimzer_g = torch.optim.Adam(itertools.chain(netg_b2a.parameters(), netg_a2b.parameters()), lr=lr)
optimzerd_a = torch.optim.Adam(netd_a.parameters(), lr)
optimzerd_b = torch.optim.Adam(netd_b.parameters(), lr)


def train():
    for epoch in range(epochs):
        avg_loss_g_a2b = AverageMeter()
        avg_loss_g_b2a = AverageMeter()
        avg_loss_d_a = AverageMeter()
        avg_loss_d_b = AverageMeter()
        min_loss_g = float('inf')
        min_loss_d = float('inf')
        for i, data in enumerate(train_loader):
            img_a, img_b = data[0].to(device), data[1].to(device)

            # update generator
            optimzer_g.zero_grad()

            fake_b = netg_a2b(img_a)
            pred_b = netd_b(fake_b)
            loss_d_b = criterionGAN(pred_b, True)

            fake_a = netg_b2a(img_b)
            pred_a = netd_a(fake_a)
            loss_d_a = criterionGAN(pred_a, True)

            recover_a = netg_b2a(fake_b)
            loss_cycle_a = criterionL1(recover_a, img_a)

            recover_b = netg_a2b(fake_a)
            loss_cycle_b = criterionL1(recover_b, img_b)

            loss_g = loss_d_a + loss_d_b + loss_cycle_a + loss_cycle_b
            loss_g.backward()

            # update discriminator  a
            optimzerd_a.zero_grad()

            pred_real_a = netd_a(img_a)
            loss_d_a_real = criterionGAN(pred_real_a, True)

            pred_fake_a = netd_a(fake_a.detach())
            loss_d_a_fake = criterionGAN(pred_fake_a, False)

            loss_a = (loss_d_a_fake + loss_d_a_real) * 0.5
            loss_a.backward()
            optimzerd_a.step()

            # update discriminator  b
            optimzerd_b.zero_grad()

            pred_real_b = netd_b(img_b)
            loss_d_b_real = criterionGAN(pred_real_b, True)

            pred_fake_b = netd_b(fake_b.detach())
            loss_d_b_fake = criterionGAN(pred_fake_b, False)

            loss_b = (loss_d_b_fake + loss_d_b_real) * 0.5
            loss_b.backward()
            optimzerd_b.step()

            # loss
            avg_loss_g_a2b.update(loss_cycle_a, loss_d_a)
            avg_loss_g_b2a.update(loss_cycle_b, loss_d_b)
            avg_loss_d_a.update(loss_a)
            avg_loss_d_b.update(loss_b)

            if i % print_freq == 0:
                print('epoch {}/{}'.format(epoch, i))
                print('loss: avg_loss_g_a2b {0} avg_loss_g_b2a{1} avg_loss_d_a{2} avg_loss_d_b{3}'
                      .format(avg_loss_g_a2b.val, avg_loss_g_b2a.val, avg_loss_d_a.avg, avg_loss_d_b.avg))
                if loss_g < min_loss_g and loss_a + loss_b < min_loss_d:
                    min_loss_g = loss_g
                    min_loss_d = loss_a + loss_b
                    torch.save((netg_a2b, netg_b2a, netd_a, netd_b), check)


if __name__ == '__main__':
    train()
