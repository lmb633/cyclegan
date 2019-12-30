from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from os.path import join

std = torch.Tensor([0.229, 0.224, 0.225])
mean = torch.Tensor([0.485, 0.456, 0.406])


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, train_or_test='train'):
        super(DatasetFromFolder, self).__init__()
        self.root = image_dir
        self.roota = join(self.root, train_or_test) + 'A'
        self.rootb = join(self.root, train_or_test) + 'B'
        self.image_a = [x for x in listdir(self.roota)]
        self.image_b = [x for x in listdir(self.rootb)]

    def __getitem__(self, index):
        a = Image.open(join(self.roota, self.image_a[index])).convert('RGB')
        b = Image.open(join(self.rootb, self.image_b[index])).convert('RGB')
        a = a.resize((112, 112), Image.BICUBIC)
        b = b.resize((112, 112), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        a = transforms.Normalize(mean, std)(a)
        b = transforms.Normalize(mean, std)(b)
        return a, b

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    dataset = DatasetFromFolder('data')
    # print(dataset[0])
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in loader:
        print(data[0].shape)
        data = (data[0].squeeze().permute(1, 2, 0) * std + mean) * 255
        data = data.float().numpy().astype(np.uint8)
        image_pil = Image.fromarray(data)
        image_pil.show()
