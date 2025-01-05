import torch.optim as optim
import torch
import torch.nn as nn
from dataloader import YOLODataset
from torch.utils.data import DataLoader
from torch import transform
from loss import YoloLoss

class YOLOModel(nn.Module):
    def __init__(self, num_classes=20, grid_size=7, bbox_per_cell=2):
        super(YOLOModel, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.bbox_per_cell = bbox_per_cell
        self.output_size = bbox_per_cell * 5 + num_classes
        self.conv = nn.Conv2d(3, self.output_size, kernel_size=1)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        out = self.conv(x)
        return out.view(batch_size, self.grid_size, self.grid_size, self.output_size)

dataset = YOLODataset(root_dir="folder", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOModel(num_classes=3).to(device)
criterion = YoloLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
