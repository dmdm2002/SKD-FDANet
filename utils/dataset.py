import torch.fft
import torchvision.utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import PIL.Image as Image
import glob
import re

from utils.transform import transform_handler


class InjectionAugDataset(Dataset):
    def __init__(self, fake_data_path, live_data_path, injection_data_path, ori_trans, aug_trans):
        super().__init__()
        self.injection_data_path = injection_data_path

        fake_images = glob.glob(f"{fake_data_path}/*")
        live_images = glob.glob(f"{live_data_path}/*/*")

        # print(f'path: {fake_data_path}')
        # print(fake_images)

        self.images = []
        for i in range(len(fake_images)):
            name = fake_images[i].split("\\")[-1]
            self.images.append([fake_images[i], 0, name])

        for i in range(len(live_images)):
            name = live_images[i].split("\\")[-1]
            self.images.append([live_images[i], 1, name])

        self.ori_trans = ori_trans
        self.aug_trans = aug_trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx][0])
        label = self.images[idx][1]

        image_o = self.ori_trans(image)
        if label == 0:
            path = f'{self.injection_data_path}/{self.images[idx][2]}'
            image_a = self.aug_trans(Image.open(path))
        else:
            image_a = self.aug_trans(image)

        return image_o, image_a, label


class AugDataset(Dataset):
    def __init__(self, fake_data_path, live_data_path, ori_trans, aug_trans):
        super().__init__()
        fake_images = glob.glob(f'{fake_data_path}/*')
        live_images = glob.glob(f'{live_data_path}/*/*')

        self.images = []
        for i in range(len(fake_images)):
            self.images.append([fake_images[i], 0])

        for i in range(len(live_images)):
            self.images.append([live_images[i], 1])

        self.ori_trans = ori_trans
        self.aug_trans = aug_trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx][0])
        label = self.images[idx][1]

        image_o = self.ori_trans(image)
        image_a = self.aug_trans(image)

        return image_o, image_a, label


class NormalDataset(Dataset):
    def __init__(self, fake_data_path, live_data_path, ori_trans):
        super().__init__()
        fake_images = glob.glob(f'{fake_data_path}/*')
        live_images = glob.glob(f'{live_data_path}/*/*')

        self.images = []
        for i in range(len(fake_images)):
            self.images.append([fake_images[i], 0])

        for i in range(len(live_images)):
            self.images.append([live_images[i], 1])

        self.ori_trans = ori_trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.ori_trans(Image.open(self.images[idx][0]))
        label = self.images[idx][1]

        return image, label


def get_loader(train,
               image_size=224,
               crop=False,
               jitter=False,
               noise=False,
               equalize=False,
               injection=False,
               batch_size=16,
               fake_path=None,
               live_path=None,
               injection_data_path=None):
    ori_trans = transform_handler(train=False,
                                  image_size=image_size)

    if train:
        if injection:
            aug_trans = transform_handler(train=train,
                                          image_size=image_size,
                                          crop=crop,
                                          jitter=jitter,
                                          noise=noise,
                                          equalize=equalize)
            dataset = InjectionAugDataset(fake_path, live_path, injection_data_path, ori_trans, aug_trans)
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        else:
            aug_trans = transform_handler(train=train,
                                          image_size=image_size,
                                          crop=crop,
                                          jitter=jitter,
                                          noise=noise,
                                          equalize=equalize)

            dataset = AugDataset(fake_path, live_path, ori_trans, aug_trans)
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    else:
        dataset = NormalDataset(fake_path, live_path, ori_trans)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return loader

