
"""
figerprint-recognition Dataset
"""
import os
import numpy as np
import torchvision
from torchvision import transforms

import ai8x

import torch
from torch.utils.data import Dataset

class FVCDataset(Dataset):

    def __init__(self, data_file,label_file, transform=None):
        '''
        data_file: 指纹的.npy格式数据
        label_file:指纹的.npy格式标签
        '''
        # 所有图片的绝对路径
        self.datas=np.load(data_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __getitem__(self, index):
        data=self.datas[index]
        label=self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        # data=data.astype(np.float32) / 255.
        label = np.argmax(label)
        return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.datas)

def fpr_get_datasets(data, load_train=True, load_test=True):
    """
    Load the figerprint-recognition dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://github.com/ZhugeKongan/Fingerprint-Recognition-pytorch-for-mcu).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    """
    (data_dir, args) = data


    if load_train:
        data_path =data_dir+'/fpr/'+'img_train.npy'
        label_path =data_dir+'/fpr/'+'label_train.npy'

        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = FVCDataset(data_path, label_path,train_transform)
    else:
        train_dataset = None

    if load_test:
        data_path = data_dir + '/fpr/' + 'img_test.npy'
        label_path = data_dir + '/fpr/' + 'label_test.npy'

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = FVCDataset(data_path, label_path,test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'fpr',
        'input': (1, 64, 64),
        'output': ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                   'F9', 'F10'),
        'loader': fpr_get_datasets,
    },
]