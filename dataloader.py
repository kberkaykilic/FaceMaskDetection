import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class YOLODataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        label_path = os.path.join(self.label_dir, self.label_files[idx])
        with open(label_path, "r") as f:
            label_data = f.readlines()
        
        labels = []
        for line in label_data:
            parts = line.strip().split()
            labels.append([float(x) for x in parts])
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels

if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    dataset_path = "/dataset"
    dataset = YOLODataset(root_dir=dataset_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)
