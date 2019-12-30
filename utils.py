import numpy as np
import torch
from PIL import Image

from model import device
from datagen import std, mean


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
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
        fake_a = modelb2a(img_b)
        for i, fake_img in enumerate(fake_b):
            save_img(fake_img.cpu().detach(), 'images/a{0}_out.jpg'.format(i))
            save_img(img_a[i].cpu().detach(), 'images/a{0}_img.jpg'.format(i))
        for i, fake_img in enumerate(fake_a):
            save_img(fake_img.cpu().detach(), 'images/b{0}_out.jpg'.format(i))
            save_img(img_b[i].cpu().detach(), 'images/b{0}_img.jpg'.format(i))
        break


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
