import os
import numpy as np
import struct
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MnistDataset(Dataset):
    def __init__(self, split='train', root_dir='data/MNIST/raw', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # File paths
        if self.split == 'train':
            self.images_path = os.path.join(root_dir, 'train-images-idx3-ubyte')
            self.labels_path = os.path.join(root_dir, 'train-labels-idx1-ubyte')
        else:
            self.images_path = os.path.join(root_dir, 't10k-images-idx3-ubyte')
            self.labels_path = os.path.join(root_dir, 't10k-labels-idx1-ubyte')

        # Load data
        self.images, self.labels = self.load_data()

    def load_data(self):
        # Load images
        with open(self.images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

        # Load labels
        with open(self.labels_path, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform:
            img = self.transform(img)
        else:
            # Default transform to Tensor and normalize to [-1, 1]
            img = transforms.ToTensor()(img)
            img = (img * 2) - 1  # Normalize to [-1, 1]

        return img, label
