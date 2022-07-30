import os
import random
import numpy as np
import cv2
import torch
from torch.utils import data as data

from styleformer.data.transforms import augment
from styleformer.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class IFFIDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.root = opt['dataroot']
        self.data_range = opt['data_range']
        
        self.img_folders = sorted(os.listdir(self.root))
        
    def __getitem__(self, index):
        # normalization
        img_folder = os.path.join(self.root, self.img_folders[index])
        img_fname = sorted(os.listdir(img_folder))[10]
        filter_idx = random.randint(0, 15)
        label = torch.tensor(filter_idx)
        filtered_img_fname = sorted(os.listdir(img_folder))[filter_idx]

        img = cv2.imread(os.path.join(img_folder, img_fname), cv2.COLOR_BGR2RGB)
        filtered_img = cv2.imread(os.path.join(img_folder, filtered_img_fname), cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / self.data_range
        filtered_img = filtered_img.astype(np.float32) / self.data_range

        # data augment
        if self.opt['phase'] == 'train':
            # flip, rotation
            img, filtered_img = augment([img, filtered_img], self.opt['use_flip'], self.opt['use_rot'])

        # numpy -> torch
        filtered_img = torch.from_numpy(np.transpose(filtered_img, (2, 0, 1))).float()
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        data = {'lq': filtered_img, 'gt': img, 'label': label}
        
        return data

    def __len__(self):
        return len(self.img_folders)


@DATASET_REGISTRY.register()
class IFFIDatasetFilterClass(data.Dataset):
    def __init__(self, opt):
        super(IFFIDatasetFilterClass).__init__()
        self.opt = opt

        self.root = opt['dataroot']
        self.data_range = opt['data_range']
        self.num_class = opt['num_class'] # 滤镜的种类数，包括原图
        
        self.img_folders = sorted(os.listdir(self.root), key=lambda x:int(x))
        self.classes = sorted(os.listdir(os.path.join(self.root, self.img_folders[0])))

    def __getitem__(self, index):
        # normalization
        img_folder = os.path.join(self.root, self.img_folders[index])
        filter_idx = random.randint(0, self.num_class - 1)

        # one hot encode
        # label = np.zeros(self.num_class)
        # label[filter_idx] = 1
        label = torch.tensor(filter_idx)

        filtered_img_fname = sorted(os.listdir(img_folder))[filter_idx]

        filtered_img = cv2.imread(os.path.join(img_folder, filtered_img_fname), cv2.COLOR_BGR2RGB)

        filtered_img = filtered_img.astype(np.float32) / self.data_range

        # data augment
        if self.opt['phase'] == 'train':
            # flip, rotation
            filtered_img = augment([filtered_img], self.opt['use_flip'], self.opt['use_rot'])

        # numpy -> torch
        filtered_img = torch.from_numpy(np.transpose(filtered_img, (2, 0, 1))).float()
        # label = torch.from_numpy(label).long()

        data = {'lq': filtered_img, 'gt': label}
        
        return data

    def __len__(self):
        return len(self.img_folders)
