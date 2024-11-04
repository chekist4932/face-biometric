import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

from src.config import BASE_DIR


class FaceDataset(Dataset):
    def __init__(self, image_dir: str):
        self.root_dir = image_dir
        self.image_paths: list = self._get_image_paths()

        self.keys = torch.load(BASE_DIR / 'models' / 'keys.pth', weights_only=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)

        label = self._get_label(image_path)

        return image, label

    def _get_image_paths(self) -> list:
        data = []
        categories = os.listdir(self.root_dir)
        for image_class in categories:

            full_path = os.path.join(self.root_dir, image_class)
            images = os.listdir(full_path)
            for image_name in images:
                im_path = os.path.join(full_path, image_name)
                data.append(im_path)
        return data

    def _get_label(self, image_path):
        label = os.path.basename(os.path.dirname(image_path))
        label = self.keys[label]
        return label


dataset_dir = BASE_DIR / 'dataset'

train_set = FaceDataset(dataset_dir / 'train')
train_loader = DataLoader(train_set, batch_size=15, shuffle=True)

test_set = FaceDataset(dataset_dir / 'test')
test_loader = DataLoader(train_set, batch_size=15, shuffle=True)
