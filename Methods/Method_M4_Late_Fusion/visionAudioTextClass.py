#Author: Jessica Kick

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
from torchvggish import vggish, vggish_input

from scripts_packages.resultsFileManagement import * #load/save pickle, plot results
from scripts_packages.metricLoss import * #mean average accuracy, average accuracy, mse per trait

from vggish import VGGish
from transformers import BertConfig, BertModel, BertTokenizer

from architectures.r2plus1finetuning import visionClass
from architectures.textFinetuning import textFinetuning as textClass
from architectures.audioClass import audioConcat as audioClass
#from visionAudioTextClassFinetuned import visionAudioTextConcat as VATClass
from architectures.visionAudioTextClassHiddenLayer import visionAudioTextConcat as VATClass

class visionAudioTextConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        #audio_model = VGGish(),
        #vision_model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True).to(torch.device('cuda:0')),
        visionParameters = r'./checkpoint/vision_parameters_finetuning.pth',
        textParameters = r'./checkpoint/text_parameters_finetuning.pth',
        audioParameters = r'./checkpoint/audio_parameters_finetuning.pth',
        #vatParameters = r'./checkpoint/visionAudioText_parameters_finetuning.pth',
        vatParameters = r'./checkpoint/visionAudioTextHidden_finetuning.pth',
        ):
        
        super(visionAudioTextConcat, self).__init__()

        self.text_module = textClass()
        self.text_module.load_state_dict(torch.load(textParameters))

        self.vision_module = visionClass()
        self.vision_module.load_state_dict(torch.load(visionParameters))

        self.audio_module = audioClass()
        self.audio_module.load_state_dict(torch.load(audioParameters))

        self.vat_module = VATClass()
        self.vat_module.load_state_dict(torch.load(vatParameters))

        #Freeze early layers
        for param in self.vision_module.parameters():
            param.requires_grad = False
        
        for param in self.text_module.parameters():
           param.requires_grad = False
        
        for param in self.audio_module.parameters():
            param.requires_grad = False

        for param in self.vat_module.parameters():
            param.requires_grad = False
        
        self.audio_module.eval()
        
        self.audio_module = self.audio_module.to(torch.device('cuda:0'))
        
        #set to inference mode
        self.vision_module.eval()
        self.vision_module = self.vision_module.to(torch.device('cuda:0'))
        self.text_module.eval()
        self.text_module = self.text_module.to(torch.device('cuda:0'))

        self.vat_module.eval()
        self.vat_module = self.vat_module.to(torch.device('cuda:0'))
        
    def forward(self, image, audio, text): #please give us input already embedded vggish
        #audioVggishEmbedding = self.audionet.forward(audio)
        audio_res = self.audio_module(audio)
        vision_res = self.vision_module(image)
        #text_features = torch.nn.functional.relu(self.text_module(**text).pooler_output)
        text_res = self.text_module(text)
        vat_res = self.vat_module(image, audio, text)

        mean_pred_avt = torch.empty(len(vision_res), 5).to(torch.device('cuda:0'))

        #---uncomment / comment for late fusion strategies of modalities ---
        #mean_pred_avt = torch.cat((vision_res, audio_res), 0)
        for i in range(len(vision_res)):
            vis = torch.unsqueeze(vision_res[i], 0)
            aud = torch.unsqueeze(audio_res[i], 0)
            tex = torch.unsqueeze(text_res[i], 0)
            vat = torch.unsqueeze(vat_res[i], 0)
            # conc = torch.cat((vis, aud, tex), 0)
            conc = torch.cat((vis, vat, aud, tex), 0)
            mean = torch.mean(conc, 0)
            mean = torch.unsqueeze(mean, 0)
            mean_pred_avt[i] = mean #torch.cat((mean_pred_avt, mean), 0)

        #print('mean: ', mean_pred_avt)
        #print('Text: ', text_res

        return mean_pred_avt

class visionAudioTextClass():
    def __init__(self, optimizer='sgd', lossfnct='mse', model='inst', learningrate=1e-5, device='cuda'):

        self.net = visionAudioTextConcat()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if(optimizer == 'sgd'):
            self.optim = torch.optim.SGD(self.net.parameters(), lr=learningrate, momentum=0.9)
        
        self.criterion = nn.MSELoss()
        self.classes = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        
        self.net.to(self.device)
    
    def test(self, val_dl, epoch):
        #evaluation
        self.net.eval()
        loss_val = 0.0
        correct = 0
        mseClassEpochVal = torch.zeros(1, len(self.classes)).to(self.device)
        traitAccPerBatch = torch.zeros(1, len(self.classes)).to(self.device)
        total = 0
    
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for batch in val_dl:
                audioBatch = batch['audio'] #.to(self.device)
                textBatch = batch['transcription']
                inputTextToken = self.tokenizer(textBatch, padding="max_length", truncation=True, return_tensors="pt")
                inputAudio, inputImage, inputText, labels = audioBatch.to(self.device), batch['image'].to(self.device), inputTextToken.to(self.device), batch['groundtruth'].float().to(self.device)
                #forward + backward + optimize
                pred = self.net(inputImage, inputAudio, inputText)
                #loss mse
                loss = self.criterion(pred, labels)
                lossPerClass = traitsMSE(pred, labels)

                loss_val += loss.item()
                mseClassEpochVal = torch.add(mseClassEpochVal, lossPerClass)

                #Average Accuracy
                traitsAcc = traitsAverageAccuracy(pred, labels)
                traitAccPerBatch = torch.add(traitAccPerBatch, traitsAcc)
    
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(pred.data, 1)
                _, labelmax = torch.max(labels, 1) #groundtruth max
                
                total += labels.size(0)
                correct += (predicted == labelmax).sum().item()
        
        print('Average Accuracy per Trait: ', traitAccPerBatch / len(val_dl))
        meanAvgAccPerEpoch = meanAverageAccuracy(traitAccPerBatch / len(val_dl), len(self.classes))
        print('Mean Average Accuracy: ', meanAvgAccPerEpoch)
        print('Epoch: {}, Loss Test: {}'.format(epoch + 1, loss_val / len(val_dl)))       
        print('Epoch: {}, Loss per Class Test: {}'.format(epoch + 1, torch.div(mseClassEpochVal, len(val_dl))))    