import torch
import torch.nn as nn


class visionClass(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        vision_model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True),
        dropout_p = 0.2,
        ):
        super(visionClass, self).__init__()
        self.vision_module = vision_model
        self.vision_module.fc = nn.Linear(512, num_classes, bias=True)
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, image):
        logits = self.vision_module(image)
        pred = torch.nn.functional.sigmoid(logits)
        
        return pred
