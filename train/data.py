import os
import h5py
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
# from torchvision import transforms
import transforms
import torch


class MiniPlace(Dataset):
    def __init__(self, data_path, split, augment=True):
        file_path = os.path.join(data_path, 'miniplaces_256_{}.h5'.format(split))
        self.dataset = h5py.File(file_path)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        transform = [
            transforms.Scale(256),
            transforms.RandomCrop(224)]

        if augment:
            transform.extend([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            ])

        transform += [
            transforms.ToTensor(),
            self.normalize]

        self.preprocess = transforms.Compose(transform)

        self.split = split
        if split != 'test':
            self.labels = np.array(self.dataset['labels'])

    def __getitem__(self, index):
        image = self.dataset['images'][index]
        img_tensor = self.preprocess(Image.fromarray(image))

        if self.split == 'test':
            return img_tensor, index

        label = self.labels[index]
        label_tensor = torch.LongTensor(np.array([label]).astype(int))

        return img_tensor, label_tensor

    def __len__(self):
        return self.dataset['images'].shape[0]
