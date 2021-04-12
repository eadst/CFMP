# -*- coding: utf-8 -*-
# @Date    : April 11, 2021
# @Author  : XD
# @Blog    ：eadst.com


from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms


# Data augmentation
aid_aug_train = transforms.Compose([

            transforms.RandomCrop(200),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.Resize(200),
            transforms.ToTensor(),
            transforms.Normalize((0.3695, 0.4102, 0.3992), (0.1888, 0.1914, 0.2115))
        ])

aid_aug_test = transforms.Compose([

            transforms.RandomCrop(230),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.Resize(230),
            transforms.ToTensor(),
            transforms.Normalize((0.3695, 0.4102, 0.3992), (0.1888, 0.1914, 0.2115))
        ])


class PreprocessData(Dataset):
    def __init__(self, data_root, data_name, transform=True, mode='train', format='root'):
        if format == 'ultimate':
            self.data = data_root
        else:
            self.data = torchvision.datasets.ImageFolder(root=data_root)
        self.name = data_name
        self.transform = transform
        self.mode = mode

    def __getitem__(self, idx):
        image, label = self.data[idx]
        data_aug_list = [(aid_aug_train, aid_aug_test)]
        if self.transform:
            data_aug = data_aug_list[0]
            if self.mode == 'train':
                image = data_aug[0](image)
            else:
                image = data_aug[1](image)
        return image, label

    def __len__(self):
        return len(self.data)
