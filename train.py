import torch
from torch import nn
from torch.utils.data import DataLoader
from datagen import DatasetFromFolder
import itertools
import os
from model import G_net, D_net, PatchLoss, device
from utils import AverageMeter, visualize, weights_init_normal, clip_weight

root = 'data/selfie2anime'

if_train_d = False
d_train_freq = 200
clip = 0.01
print_freq = 200
weight = 10
epochs = 1000
lr = 0.0002
batch_size = 1
test_batch_size = 1
input_channel = 3
output_channel = 3
ngf = 64
ndf = 64
g_layer = 9
d_layer = 4
check = 'best_checkpoint.tar'

train_set = DatasetFromFolder(root, 'train')
train_loader = DataLoader(train_set, batch_size, True)

test_set = DatasetFromFolder(root, 'test')
test_loader = DataLoader(test_set, 1, True)

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
    netd_a = D_net(input_channel, ndf, d_layer).to(device)
    netd_b = D_net(input_channel, ndf, d_layer).to(device)

criterionGAN = PatchLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
# criterionMSE = nn.MSELoss().to(device)

optimzer_g = torch.optim.SGD(itertools.chain(netg_b2a.parameters(), netg_a2b.parameters()), lr=lr)
optimzerd_a = torch.optim.SGD(netd_a.parameters(), lr)
optimzerd_b = torch.optim.SGD(netd_b.parameters(), lr)
if not os.path.exists(check):
    print('init param')
    weights_init_normal(optimzer_g)
    weights_init_normal(optimzerd_a)
    weights_init_normal(optimzerd_b)


def train():
    for epoch in range(epochs):
        avg_loss_g_a2b = AverageMeter()
        avg_loss_g_b2a = AverageMeter()
        avg_loss_g = AverageMeter()
        avg_loss_d_a = AverageMeter()
        avg_loss_d_b = AverageMeter()
        min_loss_g = float('inf')
        min_loss_d = float('inf')
        for i, data in enumerate(train_loader):
            img_a, img_b = data[0].to(device), data[1].to(device)

            #### update generator
            optimzer_g.zero_grad()
            # identity loss
            # img_b_fake = netg_a2b(img_b)
            # loss_id_b = criterionL1(img_b, img_b_fake)
            # img_a_fake = netg_b2a(img_a)
            # loss_id_a = criterionL1(img_a, img_a_fake)
            # gan loss
            fake_b = netg_a2b(img_a)
            pred_b = netd_b(fake_b)
            loss_d_b = criterionGAN(pred_b, True)

            fake_a = netg_b2a(img_b)
            pred_a = netd_a(fake_a)
            loss_d_a = criterionGAN(pred_a, True)

            # cycle loss
            recover_a = netg_b2a(fake_b)
            loss_cycle_a = criterionL1(recover_a, img_a) * 10

            recover_b = netg_a2b(fake_a)
            loss_cycle_b = criterionL1(recover_b, img_b) * 10

            loss_g = loss_d_a + loss_d_b + loss_cycle_a + loss_cycle_b
            if i % print_freq == 0:
                print('generator loss ', loss_d_a.data, loss_d_b.data, loss_cycle_a.data, loss_cycle_b.data)
                # print('generator loss ', loss_g.data)

            loss_g.backward()
            optimzer_g.step()

            if (i + 1) % d_train_freq == 0:
                #### update discriminator  a
                optimzerd_a.zero_grad()

                pred_real_a = netd_a(img_a)
                loss_d_a_real = criterionGAN(pred_real_a, True)

                pred_fake_a = netd_a(fake_a.detach())
                loss_d_a_fake = criterionGAN(pred_fake_a, False)

                loss_a = (loss_d_a_fake + loss_d_a_real) * 0.5
                # print('discriminator loss a ', loss_d_a_fake, loss_d_a_real)
                if if_train_d:
                    loss_a.backward()
                    optimzerd_a.step()
                    clip_weight(optimzerd_a, clip)

                #### update discriminator  b

                optimzerd_b.zero_grad()

                pred_real_b = netd_b(img_b)
                loss_d_b_real = criterionGAN(pred_real_b, True)

                pred_fake_b = netd_b(fake_b.detach())
                loss_d_b_fake = criterionGAN(pred_fake_b, False)

                loss_b = (loss_d_b_fake + loss_d_b_real) * 0.5
                # print('discriminator loss b ', loss_d_b_fake, loss_d_b_real)
                if if_train_d:
                    loss_b.backward()
                    optimzerd_b.step()
                    clip_weight(optimzerd_b, clip)
                avg_loss_d_a.update(loss_a)
                avg_loss_d_b.update(loss_b)

            # loss
            avg_loss_g_a2b.update(loss_d_b)
            avg_loss_g_b2a.update(loss_d_a)
            avg_loss_g.update(loss_g)

            if (i + 1) % print_freq == 0:
                print('epoch {0} {1}/{2}'.format(epoch, i, train_loader.__len__()))
                print('loss: avg_loss_g {0:.3f} avg_loss_d_a {1:.3f} avg_loss_d_b {2:.3f} avg_loss_g_d_a {3:.3f} avg_loss_g_d_b {4:.3f}'
                      .format(avg_loss_g.val, avg_loss_d_a.avg, avg_loss_d_b.avg, avg_loss_g_a2b.avg, avg_loss_g_b2a.avg))
                if loss_g < min_loss_g and loss_a + loss_b < min_loss_d:
                    min_loss_g = loss_g
                    min_loss_d = loss_a + loss_b
                    torch.save((netg_a2b, netg_b2a, netd_a, netd_b), check)

                visualize(netg_a2b, netg_b2a, test_loader)


if __name__ == '__main__':
    train()
