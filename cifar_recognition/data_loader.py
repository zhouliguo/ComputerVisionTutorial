import glob
import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, training):
        self.training = training
        self.image_paths = []
        self.labels = []
        
        for i in range(10):
            image_paths = glob.glob(os.path.join(path,str(i),'*.png'), recursive=True)
            self.image_paths = self.image_paths+image_paths
            labels = np.ones(len(image_paths))*i
            self.labels = np.append(self.labels, labels)
        
        self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.training:   # 数据增强
            if random.random() < 0.5:
                image = np.fliplr(image)
                image = np.ascontiguousarray(image)

        image = image.transpose((2, 0, 1))  # HWC to CHW

        image = torch.from_numpy(image)/255.0

        return image, label