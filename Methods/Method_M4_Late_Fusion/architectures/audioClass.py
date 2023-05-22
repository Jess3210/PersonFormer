from torchvision import models
import torch
from torchvision import transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timeit import default_timer as timer
import torch.optim.lr_scheduler as lrscheduler
import torchvision.models as models
from fire import Fire 
#from dataloader_ChaLearn import *
from sklearn.model_selection import train_test_split
import torchvision.models as models
import pickle
import os
from torchvggish import vggish, vggish_input

from resultsFileManagement import * #load/save pickle, plot results
from metricLoss import * #mean average accuracy, average accuracy, mse per trait

#from torchvggish import vggish, vggish_input
from torchvggish import vggish_input
from vggish import VGGish
from transformers import BertConfig, BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class audioConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        audio_model = VGGish(),
        #vision_model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True),
        #text_model = BertModel.from_pretrained("bert-base-uncased"),
        #language_feature_dim,
        #vision_feature_dim,
        audio_output=128,
        # fusion_audio_vision=512,
        # fusion_output_size=1280,  #768+512
        # fusion_audio_text = 768,
        lstm_size = 128,
        num_layers_lstm = 1,
        dropout_p = 0.2
        ):

        super(audioConcat, self).__init__()
        #audio_model.to('cuda:0')
        #self.audio_module = audio_model.to('cuda:0')
        
        #self.audio_module = self.audio_module

        # Freeze early layers
        # for param in self.audio_module.parameters():
        #     param.requires_grad = False

        self.audio_module = audio_model.to(torch.device('cuda:0'))

        #audio_model.load_state_dict(torch.load('pytorch_vggish.pth'))
        self.audio_module.load_state_dict(torch.load('pytorch_vggish.pth'))

        # for param in self.audio_module.parameters():
        #     param.requires_grad = False
        
        # self.audio_module.eval()
        
        self.audio_module = self.audio_module.to(torch.device('cuda:0'))

        #print('Device here: ', self.audio_module.device)
        
        #self.audio_module.eval()

        # self.audio_layer = torch.nn.Linear(
        #         in_features=128,
        #         out_features=100
        # )

        # self.num_layers_lstm = num_layers_lstm
        # self.hidden_dim = lstm_size
        
        # self.vision_module = vision_model
        # self.text_module = text_model

        # Freeze early layers
        # for param in self.vision_module.parameters():
        #     param.requires_grad = False
        
        # for param in self.text_module.parameters():
        #     param.requires_grad = False
        
        #set to inference mode
        # self.vision_module.eval()
        # self.text_module.eval()

        # self.vision_module.fc = nn.Linear(512, 512, bias=True)

        # self.lstm_module = torch.nn.LSTM(
        #         input_size=lstm_size,
        #         hidden_size=lstm_size,
        #         num_layers=num_layers_lstm,
        #         batch_first=True,
        # )

        # output_audio = self.dropout(
        #     self.fc1(audioOneSecond_features)
        # )
        # logits = self.fc2(output_audio)
        self.fc1 = torch.nn.Linear(
            in_features=audio_output, 
            out_features=audio_output
        )
        self.fc2 = torch.nn.Linear(
            in_features=audio_output, 
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, audioOneSecond): #please give us input already embedded vggish
        # audio_features = torch.nn.functional.relu(
        #     self.audio_module(audio)
        # )
        audio_features = self.audio_module(audioOneSecond)
        
        #print('shape heere: ', audioOneSecond.shape)
        
        #audio_features = torch.empty((len(audioOneSecond), 15, 128)).to(torch.device('cuda:0'))
        #for audio in audioOneSecond:
            # vggEmb = self.audio_module(audio)
            # audio_features = torch.add(audio_features, vggEmb)

        #audiolen = [audiosize for audiosize in audiolen]   # get the length of each sequence in the batch
        
        output_audio = self.dropout(
            torch.nn.functional.relu(
            self.fc1(audio_features)
        )
        )
    
        logits = self.fc2(output_audio)
        pred = torch.nn.functional.sigmoid(logits)
        
        return pred
    
    def init_hidden(self, batch_size):
        #source: https://discuss.pytorch.org/t/lstm-init-hidden-to-gpu/81441
        weight = next(self.lstm_module.parameters()).data
        hidden = (weight.new(self.num_layers_lstm, batch_size, self.hidden_dim).zero_(),
                weight.new(self.num_layers_lstm, batch_size, self.hidden_dim).zero_())
        return hidden
    
    # def padding(self, audiopersecond):
    #     audioSecond = audiopersecond
    #     padd = torch.zeros(1, len(audioSecond[0])) # dim: 1, 128
    #     diffLen = 15 - len(audioSecond)
        
    #     if diffLen > 0:
    #         for i in range(diffLen):
    #             audioSecond = torch.cat((audioSecond, padd), 0) 
    #     return audioSecond
