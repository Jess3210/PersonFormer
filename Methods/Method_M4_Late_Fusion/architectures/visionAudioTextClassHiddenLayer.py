#Author: Jessica Kick

import torch
import numpy as np
import torch.nn as nn

from torchvggish import vggish, vggish_input
from architectures.vggish import VGGish
from transformers import BertConfig, BertModel, BertTokenizer

from architectures.r2plus1finetuning import visionClass
from architectures.textFinetuning import textFinetuning as textClass
from architectures.vggish import VGGish

class visionAudioTextConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        audio_model = VGGish(torch.load('./checkpoint/pytorch_vggish.pth')), #no pretrained weights: VGGish(None)
        vision_model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True).to(torch.device('cuda:0')),
        text_model = BertModel.from_pretrained("bert-base-uncased").to(torch.device('cuda:0')),
        fusion_audio_vision=512,
        fusion_output_size=1280,  #768+512
        fusion_output_size2=512,
        dropout_p = 0.2,
        ):
        
        super(visionAudioTextConcat, self).__init__()

        self.audio_module = audio_model
        self.vision_module = vision_model

        self.text_module = text_model

        #Freeze early layers
        for param in self.vision_module.parameters():
            param.requires_grad = False
        
        for param in self.audio_module.parameters():
            param.requires_grad = False
        
        self.audio_module.eval()
        
        self.audio_module = self.audio_module.to(torch.device('cuda:0'))
        
        #set to inference mode
        self.vision_module.eval()
        #self.text_module.eval()
        self.text_module = self.text_module.to(torch.device('cuda:0'))

        self.audio_features = torch.nn.Linear(
                in_features=128,
                out_features=100
        )
        
        self.vision_module.fc = nn.Linear(512, 512, bias=True)

        self.text_module.fc = nn.Linear(768, 768, bias=True)
        
        self.fusion_av = torch.nn.Linear(
            in_features=(100 + 512), 
            out_features=fusion_audio_vision
        )
        self.fusion_avt = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=fusion_output_size
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=fusion_output_size2
        )
        self.fc2 = torch.nn.Linear(
            in_features=fusion_output_size2, 
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)
        self.dropout2 = torch.nn.Dropout(dropout_p)
        
    def forward(self, image, audio, text): #please give us input already embedded vggish
        audio_vgg = self.audio_module(audio)
        audio_features = torch.nn.functional.relu(
            self.audio_features(audio_vgg)
        )
        #print(audio_features.shape)
        image_features = torch.nn.functional.relu(
            self.vision_module(image)
        )
        text_features = torch.nn.functional.relu(self.text_module(**text).pooler_output)
        
        combined_av = torch.cat(
            [audio_features, image_features], dim=1
        )
        
        fused_av = torch.nn.functional.relu(
            self.fusion_av(combined_av)
        )
        #Second: Combine (A+V) + T
        combined_avt = torch.cat(
            [fused_av, text_features], dim=1
        )
        fused_avt = self.dropout(
            torch.nn.functional.relu(
            self.fusion_avt(combined_avt)
            )
        )

        layer2 = self.dropout2(
            torch.nn.functional.relu(
            self.fc(fused_avt)
            )
        )

        logits = self.fc2(layer2)
        pred = torch.nn.functional.sigmoid(logits)
        
        return pred