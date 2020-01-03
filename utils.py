import numpy as np
import torch
from PIL import Image

from model import device
from datagen import std, mean
from torch.nn import init


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_tensor = (image_tensor.squeeze().permute(1, 2, 0) * std + mean) * 255
    image_tensor = image_tensor.float().numpy().clip(0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image_tensor)
    image_pil.save(filename)
    # print("Image saved as {}".format(filename))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, hmean, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'hmean': hmean,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


def visualize(modela2b, modelb2a, dataloader):
    for _, data in enumerate(dataloader):
        img_a, img_b = data[0].to(device), data[1].to(device)
        fake_b = modela2b(img_a)
        recover_a = modelb2a(fake_b)
        fake_a = modelb2a(img_b)
        recover_b = modela2b(fake_a)
        for i, fake_img in enumerate(fake_b):
            save_img(fake_img.cpu().detach(), 'images/a{0}_out.jpg'.format(i))
            save_img(recover_a[i].cpu().detach(), 'images/a{0}_recover.jpg'.format(i))
            save_img(img_a[i].cpu().detach(), 'images/a{0}_img.jpg'.format(i))
        for i, fake_img in enumerate(fake_a):
            save_img(fake_img.cpu().detach(), 'images/b{0}_out.jpg'.format(i))
            save_img(recover_b.cpu().detach(), 'images/b{0}_recover.jpg'.format(i))
            save_img(img_b[i].cpu().detach(), 'images/b{0}_img.jpg'.format(i))
        break


def weights_init_normal(optimizer):
    for group in optimizer.param_groups:
        for param in group['params']:
            init.normal_(param.data)


def clip_weight(optimizer, weight_clip=0.01):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.data.clamp_(-weight_clip, weight_clip)
