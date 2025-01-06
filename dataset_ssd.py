import os
import torch
import cv2
from torch.utils.data import Dataset


class SSD_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(320, 320)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')

        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found at {self.images_dir}")

        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.png', '.txt'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        image = torch.FloatTensor(image / 255.0).permute(2, 0, 1)

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    labels.append(int(data[0]))
                    x, y, w, h = map(float, data[1:])
                    xmin = (x - w / 2) * self.target_size[0]
                    ymin = (y - h / 2) * self.target_size[1]
                    xmax = (x + w / 2) * self.target_size[0]
                    ymax = (y + h / 2) * self.target_size[1]
                    boxes.append([xmin, ymin, xmax, ymax])

        if not boxes:
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        return image, torch.FloatTensor(boxes), torch.LongTensor(labels)
