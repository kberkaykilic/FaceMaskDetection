import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        pred_box = predictions[..., :4]
        pred_obj = predictions[..., 4]
        pred_class = predictions[..., 5:]

        target_box = targets[..., :4]
        target_obj = targets[..., 4]
        target_class = targets[..., 5:]

        box_loss = self.lambda_coord * self.mse(pred_box, target_box)
        obj_loss = self.bce(pred_obj, target_obj)
        noobj_loss = self.lambda_noobj * self.bce(pred_obj, target_obj * 0)
        class_loss = self.ce(pred_class, target_class.argmax(-1))

        total_loss = box_loss + obj_loss + noobj_loss + class_loss
        return total_loss
