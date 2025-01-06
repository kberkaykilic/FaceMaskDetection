import torch
import torch.nn as nn
from dataset_ssd import SSD_Dataset
from SSD import SSDLite
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from tqdm import tqdm
import shutil

def collate_fn(batch):
    images = []
    boxes = []
    labels = []

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    images = torch.stack(images, 0)

    return images, boxes, labels

def train_ssd(num_epochs=80, batch_size=16, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SSDLite(num_classes=3).to(device)

    train_dataset = SSD_Dataset('face-mask-detection/train')
    val_dataset = SSD_Dataset('face-mask-detection/val')

    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=2,
                           collate_fn=collate_fn)

    loc_criterion = nn.MSELoss()
    conf_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf')
    }

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for batch_idx, (images, boxes_list, labels_list) in enumerate(train_bar):
            images = images.to(device)
            batch_size = images.size(0)

            loc_preds, conf_preds = model(images)

            loc_preds = loc_preds.view(batch_size, -1, 4)  # [batch_size, num_boxes, 4]
            conf_preds = conf_preds.view(batch_size, -1, 3)  # [batch_size, num_boxes, num_classes]

            batch_loss = 0
            for i in range(batch_size):
                if len(boxes_list[i]) > 0:
                    cur_loc_preds = loc_preds[i]
                    cur_conf_preds = conf_preds[i]

                    cur_boxes = boxes_list[i].to(device)
                    cur_labels = labels_list[i].to(device)

                    num_boxes = len(cur_boxes)
                    loc_loss = loc_criterion(
                        cur_loc_preds[:num_boxes],
                        cur_boxes
                    )

                    conf_loss = conf_criterion(
                        cur_conf_preds[:num_boxes],
                        cur_labels
                    )

                    loss = loc_loss + conf_loss
                    batch_loss += loss

            if batch_size > 0:
                batch_loss = batch_loss / batch_size

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()
                train_bar.set_postfix({'loss': train_loss/(batch_idx+1)})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')

        with torch.no_grad():
            for images, boxes_list, labels_list in val_bar:
                images = images.to(device)
                batch_size = images.size(0)

                loc_preds, conf_preds = model(images)

                loc_preds = loc_preds.view(batch_size, -1, 4)
                conf_preds = conf_preds.view(batch_size, -1, 3)

                batch_loss = 0
                for i in range(batch_size):
                    if len(boxes_list[i]) > 0:
                        cur_loc_preds = loc_preds[i]
                        cur_conf_preds = conf_preds[i]

                        cur_boxes = boxes_list[i].to(device)
                        cur_labels = labels_list[i].to(device)

                        num_boxes = len(cur_boxes)
                        loc_loss = loc_criterion(
                            cur_loc_preds[:num_boxes],
                            cur_boxes
                        )
                        conf_loss = conf_criterion(
                            cur_conf_preds[:num_boxes],
                            cur_labels
                        )

                        batch_loss += loc_loss + conf_loss

                if batch_size > 0:
                    batch_loss = batch_loss / batch_size
                    val_loss += batch_loss.item()
                    val_bar.set_postfix({'loss': val_loss/len(val_loader)})

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            torch.save(model.state_dict(), 'ssd_model.pth')

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, f'ssd_checkpoint_epoch_{epoch+1}.pth')

    return model, history