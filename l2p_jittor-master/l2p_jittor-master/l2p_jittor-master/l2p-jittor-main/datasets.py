# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

from math import e
import pickle
import random

# 这块Jittor的Subset功能不完善，故使用torch.utils.data.dataset.Subset
from torch.utils.data.dataset import Subset


# Jittor这部分功能不完善，故这一块使用torchvision的datasets
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

from jittor.dataset import DataLoader, Dataset
import gc


class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

import jittor as jt
from jittor.dataset import Dataset
from tqdm import tqdm


class JittorDataset(Dataset):
    def __init__(self, pytorch_dataset):
        super().__init__()
      
        # 检查是否是 Subset
        if hasattr(pytorch_dataset, "dataset"):
            # 如果是 Subset，获取原始数据
            actual_dataset = pytorch_dataset.dataset
            indices = pytorch_dataset.indices if hasattr(pytorch_dataset, "indices") else range(len(pytorch_dataset))
          
            # 提取所有数据 - 这样会保持变换后的尺寸
            data_list = []
            targets_list = []
            for idx in indices:
                img, label = actual_dataset[idx]  # 获取经过变换的数据
                data_list.append(np.array(img))
                targets_list.append(label)
          
            self.data = jt.array(np.stack(data_list))
            self.targets = jt.array(np.array(targets_list))
        else:
            # 检查数据集是否有 transforms
            if hasattr(pytorch_dataset, 'transform') and pytorch_dataset.transform is not None:
                # 如果有 transforms，通过 __getitem__ 获取变换后的数据
                print(f"Dataset has transforms, extracting transformed data to preserve input shape...")
                data_list = []
                targets_list = []
                
                # 临时保存原始 transform
                original_transform = pytorch_dataset.transform
                
                for i in range(len(pytorch_dataset)):
                    img, label = pytorch_dataset[i]  # 获取变换后的数据
                    data_list.append(np.array(img))
                    targets_list.append(label)
                
                self.data = jt.array(np.stack(data_list))
                self.targets = jt.array(np.array(targets_list))
                
            else:
                # 如果没有 transforms，使用原始方法
                dataset_name = type(pytorch_dataset).__name__
                
                if dataset_name in ['MNIST', 'FashionMNIST']:
                    self.data = jt.array(pytorch_dataset.data.numpy())
                    self.targets = jt.array(pytorch_dataset.targets.numpy())
                    
                elif dataset_name == 'CIFAR10':
                    self.data = jt.array(pytorch_dataset.data)
                    self.targets = jt.array(np.array(pytorch_dataset.targets))
                    
                elif dataset_name == 'SVHN':
                    self.data = jt.array(pytorch_dataset.data)
                    self.targets = jt.array(pytorch_dataset.labels)
                    
                elif dataset_name == 'NotMNIST':
                    if hasattr(pytorch_dataset, 'data') and hasattr(pytorch_dataset, 'targets'):
                        data = pytorch_dataset.data
                        targets = pytorch_dataset.targets
                        
                        if hasattr(data, 'numpy'):
                            data = data.numpy()
                        if hasattr(targets, 'numpy'):
                            targets = targets.numpy()
                        elif isinstance(targets, list):
                            targets = np.array(targets)
                            
                        self.data = jt.array(data)
                        self.targets = jt.array(targets)
                    else:
                        data_list = []
                        targets_list = []
                        for i in range(len(pytorch_dataset)):
                            img, label = pytorch_dataset[i]
                            data_list.append(np.array(img))
                            targets_list.append(label)
                        
                        self.data = jt.array(np.stack(data_list))
                        self.targets = jt.array(np.array(targets_list))
                else:
                    # 通用方法 - 保持变换后的形状
                    print(f"Using generic method for dataset type {dataset_name}")
                    data_list = []
                    targets_list = []
                    for i in range(len(pytorch_dataset)):
                        img, label = pytorch_dataset[i]
                        data_list.append(np.array(img))
                        targets_list.append(label)
                    
                    self.data = jt.array(np.stack(data_list))
                    self.targets = jt.array(np.array(targets_list))
      
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target
        
    def __len__(self):
        return len(self.data)
def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    else:
        # ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']'SVHN'太大了，不跑
        if args.dataset == '5-datasets':
            if args.output_dir == './output_NotMNIST':
                dataset_list = ['NotMNIST']
            elif args.output_dir == './output_CIFAR10':
                dataset_list = ['CIFAR10']
            elif args.output_dir == './output_FashionMNIST':
                dataset_list = ['FashionMNIST']
            elif args.output_dir == './output_MNIST':
                dataset_list = ['MNIST']
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle:
            random.shuffle(dataset_list)
        print(dataset_list)
    
        args.nb_classes = 0

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
        
        # 暂不考虑分布式学习：
        # print(f"Single image shape: {image.shape}") 
        if not isinstance(dataset_train, Dataset):
            dataset_train = JittorDataset(dataset_train)
        if not isinstance(dataset_val, Dataset):
            dataset_val = JittorDataset(dataset_val)
            
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,           # 开启随机打乱
            num_workers=args.num_workers,
        )
        # 验证集 DataLoader（顺序采样）
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,          # 关闭打乱
            num_workers=args.num_workers,
        )
        
        dataloader.append({'train': data_loader_train, 'val': data_loader_val})
        
        #保存dataloader, class_mask,
        
    return dataloader, class_mask

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=False, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=False, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=False, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=False, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=False, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=False, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=False, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=False, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=False, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=False, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=False, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=False, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data
    
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        print('Resize input')
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)