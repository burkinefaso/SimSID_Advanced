import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import random

class KarmaDataset(Dataset):
    def __init__(self, root, train=True, img_size=(256, 256), 
                 normalize=False, normalize_tanh=False, 
                 enable_transform=True, full=True, positive_ratio=1.0):

        self.data = []
        self.fnames = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full
        self.positive_ratio = positive_ratio

        # Transformasyonlar
        self.transforms = self._build_transforms(train, enable_transform, normalize_tanh)
        
        # Veri yükleme
        self.load_data()

    def _build_transforms(self, train, enable_transform, normalize_tanh):
        transform_list = []
        if train and enable_transform:
            transform_list.extend([
                transforms.RandomAffine(0, translate=(0.05, 0.05)),
                transforms.ToTensor()
            ])
        else:
            transform_list.append(transforms.ToTensor())
        
        if normalize_tanh:
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        return transforms.Compose(transform_list)

    def load_data(self):
        self.data = []
        self.fnames = []
        
        # TEST modu
        if not self.train:
            # NORMAL klasöründen tüm örnekler
            normal_path = os.path.join(self.root, 'NORMAL')
            if os.path.exists(normal_path):
                for item in os.listdir(normal_path):
                    img = Image.open(os.path.join(normal_path, item)).resize(self.img_size)
                    self.data.append((img, 0))
                    self.fnames.append(item)
            
            # ANOMALY klasöründen tüm örnekler
            anomaly_path = os.path.join(self.root, 'ANOMALY')
            if os.path.exists(anomaly_path):
                for item in os.listdir(anomaly_path):
                    img = Image.open(os.path.join(anomaly_path, item)).resize(self.img_size)
                    self.data.append((img, 1))
                    self.fnames.append(item)
            
            print(f'[TEST] {len(self.data)} data loaded from: {self.root}')
            return

        # TRAIN modu
        pos_items = os.listdir(os.path.join(self.root, 'NORMAL'))
        neg_items = os.listdir(os.path.join(self.root, 'ANOMALY'))
        
        num_pos = int(len(pos_items) * self.positive_ratio)
        num_neg = len(neg_items) if self.full else len(pos_items) - num_pos
        
        # Pozitif örnekler (NORMAL)
        for item in pos_items[:num_pos]:
            img = Image.open(os.path.join(self.root, 'NORMAL', item)).resize(self.img_size)
            self.data.append((img, 0))
            self.fnames.append(item)
        
        # Negatif örnekler (ANOMALY)
        for item in neg_items[:num_neg]:
            img = Image.open(os.path.join(self.root, 'ANOMALY', item)).resize(self.img_size)
            self.data.append((img, 1))
            self.fnames.append(item)
        
        print(f'[TRAIN] {len(self.data)} data loaded from: {self.root}, positive rate: {self.positive_ratio:.2f}')

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.transforms(img)[[0]]  # Gri tonlamalı tek kanal
        if self.normalize:
            img = (img - self.mean) / self.std
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)
