import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class Point_face(Dataset):
    def __init__(self, meta_path_or_dict, transform=None):
        super(Point_face, self).__init__()
        self.transform = transform
        if isinstance(meta_path_or_dict, str):
            with open(meta_path_or_dict, 'r') as f:
                self.meta = json.load(f)
        elif isinstance(meta_path_or_dict, dict):
            self.meta = meta_path_or_dict

        self.samples = list(self.meta.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        data = self.samples[item]
        img_path = data["image_path"]
        img = Image.open(img_path).convert('RGB')
        x1, y1, x2, y2 = data["bbox"]
        landmarks = np.array(data["landmarks"], dtype=np.float32)
        img_crop = img.crop((x1, y1, x2, y2))
        crop_w, crop_h = img_crop.size
        landmarks_shifted = landmarks.copy()
        landmarks_shifted[:, 0] -= x1
        landmarks_shifted[:, 1] -= y1
        landmarks_norm = landmarks_shifted.copy()
        landmarks_norm[:, 0] /= (crop_w if crop_w > 0 else 1.0)
        landmarks_norm[:, 1] /= (crop_h if crop_h > 0 else 1.0)

        landmarks_flatten = landmarks_norm.flatten()
        landmarks_tensor = torch.tensor(landmarks_flatten, dtype=torch.float32)
        img_resized = img_crop.resize((224, 224), Image.BILINEAR)
        if self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)

        return {"image": img_tensor, "label": landmarks_tensor,
                "orig_bbox": torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
                "crop_shape": torch.tensor([crop_w, crop_h], dtype=torch.float32)
                }