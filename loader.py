import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import transforms as package_transforms
import label as custom_label

from PIL import Image

def is_annotated_image(filename):
    return any(filename.endswith("_a" + extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(filepath):
    img = Image.open(filepath)
    img.putalpha(256)
    return img.convert('RGB')

def image_pairs(dataPath):
    annotated_images = [x for x in os.listdir(dataPath) if is_annotated_image(x)]
    base_image = lambda x: x[:-6] + x[-6:].replace('a', 'b')
    return [(base_image(x), x) for x in annotated_images]

class Dataset(data.Dataset):
    def __init__(self, dataPath, test=False, custom_transforms = package_transforms.Transforms()):
        super(Dataset, self).__init__()
        self.dataPath = dataPath
        self.image_pairs = image_pairs(self.dataPath) # base, annotated
        self.image_pairs = sorted(self.image_pairs, key=lambda x: x[0])

        if not test:
            self.transform = transforms.Compose([
                transforms.Lambda(custom_transforms.apply_transformations),
                transforms.Lambda(custom_label.image_to_label_transform),
                transforms.Lambda(custom_transforms.to_tensor)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(custom_label.image_to_label_transform),
                transforms.Lambda(custom_transforms.to_tensor)
            ])

    def __getitem__(self, index):
        path_b = os.path.join(self.dataPath, self.image_pairs[index][0]) 
        path_a = os.path.join(self.dataPath, self.image_pairs[index][1])

        img = default_loader(path_b)
        label = default_loader(path_a)

        img, label = self.transform((img, label))

        img = img.transpose(0, -1)

        return img, label

    def __len__(self):
        return len(self.image_pairs)