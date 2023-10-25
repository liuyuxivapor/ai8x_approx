import os
import numpy as np
import torchvision
from torchvision import transforms

import ai8x

import torch
from torch.utils.data import Dataset

class FFTDataset(Dataset):

    def __init__(self, data_file, label_file, transform=None):
        self.datas = np.load(data_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        label = np.argmax(label)
        return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.datas)

def fft_get_datasets(data, load_train=True, load_test=False):

    (data_dir, args) = data

    if load_train:
        data_path = '/home/vapor/code/AIoT/ai8x-training/data'+'/fft/'+'data.npy'
        label_path = '/home/vapor/code/AIoT/ai8x-training/data'+'/fft/'+'fft_result.npy'

        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = FFTDataset(data_path, label_path, train_transform)
    else:
        train_dataset = None

    if load_test:
        data_path = data_dir + '/fft/' + 'img_test.npy'
        label_path = data_dir + '/fft/' + 'label_test.npy'

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = FFTDataset(data_path, label_path, test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[ :1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'fft',
        'input': (1, ),
        'output': (2, ),
        'loader': fft_get_datasets,
    },
]