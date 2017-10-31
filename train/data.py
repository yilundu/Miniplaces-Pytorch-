import os
import h5py
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch


class MiniPlace(Dataset):
    def __init__(self, data_path, split):
        file_path = os.path.join(data_path, 'miniplaces_256_{}.h5'.format(split))
        self.dataset = h5py.File(file_path)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.preprocess = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize])

        self.labels = np.array(self.dataset['labels'])

    def __getitem__(self, index):
        image = self.dataset['images'][index]
        label = self.labels[index]

        img_tensor = self.preprocess(Image.fromarray(image))
        label_tensor = torch.LongTensor(np.array([label]).astype(int))

        return img_tensor, label_tensor

    def __len__(self):
        return self.dataset['images'].shape[0]
