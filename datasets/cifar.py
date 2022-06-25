import os.path as osp
import pickle
import numpy as np
import scipy.io as sio
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import transform as T
from datasets.randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler
from datasets.ctaugment import *
from PIL import Image 
import torchvision

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class TwoCropsTransform:
    """Take 2 random augmentations of one image."""

    def __init__(self,trans_weak, trans_strong):       
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]

def load_data_train(L=250, dataset='Kvasir-Capsule', dspth='./data', seed=1):
    w, h = 336, 336
    if dataset == 'Kvasir-Capsule':
        data, labels = [], []
        train_root = '/home/qsyang2/codes/ssl/datasets/Kvasir-Capsule/select_data/train_data'
        train_dataset = ImageFolderWithPaths(train_root)
        
        for entry in train_dataset:
            _tmp_data = np.array(Image.open(entry[2]).convert('RGB'))
            data.append(np.expand_dims(np.transpose(_tmp_data, (2,0,1)), axis=0))
            labels.append(np.expand_dims(entry[1], axis=0))
        data = np.concatenate(data, axis=0)      # [3560, 3, 336, 336]
        labels = np.concatenate(labels, axis=0)  # [3560]

        n_class = 10
        assert L in [int(3560*0.3), int(3560*0.2), int(3560*0.1), int(3560*0.05), int(3559*1)]   # [1068, 712, 356, 178, 3559]
       
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        data, labels = [], []
        for data_batch in datalist:
            with open(data_batch, 'rb') as fr:
                entry = pickle.load(fr, encoding='latin1')
                lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
                data.append(entry['data'])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
    
    n_labels = L // n_class
    data_x, label_x, data_u, label_u = [], [], [], []

    for i in range(n_class):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        inds_x, inds_u = indices[:n_labels], indices[n_labels:]
        data_x += [
            data[i].reshape(3, w, h).transpose(1, 2, 0)
            for i in inds_x
        ]
        label_x += [labels[i] for i in inds_x]
        data_u += [
            data[i].reshape(3, w, h).transpose(1, 2, 0)
            for i in inds_u
        ]
        label_u += [labels[i] for i in inds_u]

    return data_x, label_x, data_u, label_u


def load_data_val(dataset, dspth='./data', seed=1):
    w, h = 336, 336
    if dataset == 'Kvasir-Capsule':
        data, labels = [], []
        test_root = '/home/qsyang2/codes/ssl/datasets/Kvasir-Capsule/select_data/test_data'
        test_dataset = ImageFolderWithPaths(test_root)

        for entry in test_dataset:
            _tmp_data = np.array(Image.open(entry[2]).convert('RGB'))
            data.append(np.expand_dims(np.transpose(_tmp_data, (2,0,1)), axis=0))
            labels.append(np.expand_dims(entry[1], axis=0))
        data = np.concatenate(data, axis=0)      # [3560, 3, 336, 336]
        labels = np.concatenate(labels, axis=0)  # [3560]

    data = [
        el.reshape(3, w, h).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


def get_train_loader(dataset, batch_size, mu, n_iters_per_epoch, L, root='data', method='fixmatch', aug_type='RA', seed=1):
    if dataset == 'STL10':
        data_x, label_x, data_u, label_u, test_data, test_label = load_data_train(L=L, dataset=dataset, dspth=root, seed=seed)
    else:
        data_x, label_x, data_u, label_u = load_data_train(L=L, dataset=dataset, dspth=root, seed=seed)

    ds_x = Cifar(
        dataset=dataset,
        data=data_x,
        labels=label_x,
        mode='train_x',
        aug_type=aug_type
    ) 
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=1,
        pin_memory=True
    )
    
    ds_u = Cifar(
        dataset=dataset,
        data=data_u,
        labels=label_u,
        mode='train_u_%s'%method,
        aug_type=aug_type
    )

    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=1,
        pin_memory=True
    )

    return dl_x, dl_u


def get_val_loader(dataset, batch_size, num_workers, pin_memory=True, root='data', aug_type='RA', seed=1):
    data, labels = load_data_val(dataset, dspth=root, seed=seed)
    ds = Cifar(
        dataset=dataset,
        data=data,
        labels=labels,
        mode='test',
        aug_type=aug_type
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


class Cifar(Dataset):
    def __init__(self, dataset, data, labels, mode, aug_type):
        super(Cifar, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        w, h = 128,128 
        if dataset == 'Kvasir-Capsule':
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        
        trans_weak = T.Compose([
            T.Resize((w, h)),
            T.PadandRandomCrop(border=4, cropsize=(w, h)),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_weak_noflip = T.Compose([
            T.Resize((w, h)),
            T.PadandRandomCrop(border=4, cropsize=(w, h)),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        
        if aug_type == 'RA':
            trans_strong0 = T.Compose([
                T.Resize((w, h)),
                T.PadandRandomCrop(border=4, cropsize=(w, h)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

            trans_strong0_noflip = T.Compose([
                T.Resize((w, h)),
                T.PadandRandomCrop(border=4, cropsize=(w, h)),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])   
        
        elif aug_type == 'CTA':
            trans_strong0 = T.Compose([
                T.Resize((w, h)),
                T.PadandRandomCrop(border=4, cropsize=(w, h)),
                T.RandomHorizontalFlip(p=0.5),
                CTAugment(),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0_noflip = T.Compose([
                T.Resize((w, h)),
                T.PadandRandomCrop(border=4, cropsize=(w, h)),
                CTAugment(),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])  
            
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(w, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),        
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])                    
        
        if self.mode == 'train_x':
            self.trans = trans_weak         
        elif self.mode == 'train_u_fixmatch':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)    
        else:  
            self.trans = T.Compose([
                T.Resize((w, h)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


def compute_mean_var(dataset=None, dspth='./data'):
    data_x, label_x, data_u, label_u = load_data_train(dataset=dataset, dspth=dspth)
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)
    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))
    print('mean: ', mean)
    print('var: ', var)

    return mean, var


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)
    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class
