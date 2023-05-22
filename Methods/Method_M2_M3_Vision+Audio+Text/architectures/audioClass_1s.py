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

        self.audio_module = audio_model.to(torch.device('cuda:0'))

        #audio_model.load_state_dict(torch.load('pytorch_vggish.pth'))
        self.audio_module.load_state_dict(torch.load('pytorch_vggish.pth'))
        #self.audio_module = VGGish(torch.load('pytorch_vggish.pth'))

        # for param in self.audio_module.parameters():
        #    param.requires_grad = False
        
        self.audio_module.eval()
        
        self.audio_module = self.audio_module.to(torch.device('cuda:0'))

        self.audio_layer = torch.nn.Linear(
                in_features=audio_output,
                out_features=audio_output
        )

        self.fc = torch.nn.Linear(
            in_features=audio_output, 
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, audioOneSecond): #please give us input already embedded vggish
        #audio_features = self.audio_module(audio)
        #audio_features = self.audio_module(audioOneSecond)

        audio_features = torch.empty((len(audioOneSecond), 15, 128)).to(torch.device('cuda:0'))
        # for audio in audioOneSecond:
        #     vggEmb = self.audio_module(audio)
        #     audio_features = torch.add(audio_features, vggEmb)

        # audioOneSecond has shape (batch_size, 15, 96, 64, 1)
        # Reshape it to (batch_size * 15, 1, 96, 64)
        audio = audioOneSecond.reshape(-1, 1, 96, 64)
        #print('Shape after reshape: ', audio.shape)
        vggEmb = self.audio_module(audio)
        vggEmb = vggEmb.reshape(-1, 15, vggEmb.shape[-1])
        #print('Shape after reshape the reshape: ', vggEmb.shape)
        #audio_features = torch.add(audio_features, vggEmb)
        
        #print(vggEmb)
        #print('shape of: ', vggEmb.shape)
        audio_features = vggEmb.mean(dim=1)
        #print('Audio Features final: ', audio_features.shape)
        
        #audiolen = [audiosize for audiosize in audiolen]   # get the length of each sequence in the batch

        output_audio = self.dropout(torch.nn.functional.relu(
            self.audio_layer(audio_features)
        ))
        logits = self.fc(output_audio)
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

class audioClass():
    def __init__(self, optimizer='sgd', lossfnct='mse', model='inst', learningrate=1e-5, device='cuda', bs=32):

        self.pathRes = './audioMean'
        isExist = os.path.exists(self.pathRes)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.pathRes)
            os.makedirs(self.pathRes + '/checkpoint')

        self.net = audioConcat()
        #self.netVggish = vggish()
        #self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.batch_size = bs

        #print(self.net)
       
        if(optimizer == 'sgd'):
            self.optim = torch.optim.SGD(self.net.parameters(), lr=learningrate, momentum=0.9)
        
        self.criterion = nn.MSELoss()
        self.classes = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        checkpointLoader = torch.load('audioMean/checkpoint/audio_parameters_checkpoint.pt')
        self.net.load_state_dict(checkpointLoader['model_state_dict'])
        self.optim.load_state_dict(checkpointLoader['optimizer_state_dict'])

        #send optimizer to GPU
        self.optimizer_to(self.optim, self.device)
        
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5, factor=0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5, factor=0.1)

        # Freeze early layers
        #for param in self.netVggish.parameters():
        #    param.requires_grad = False
        
        #n_inputs = self.net.fc.in_features
        #self.net.fc = nn.Sequential(nn.Linear(512, len(self.classes), bias=True), nn.Sigmoid())
        
        self.net.to(self.device)
        #self.hidden = self.net.init_hidden(self.batch_size)

        #self.netVggish.to('cpu')
        #self.netVggish.eval()

        #print('Network device: ', next(self.net.parameters()).device)
        
        # self.loss_values_train = []
        # self.loss_values_val = []
        # self.loss_values_test = []
        
        # self.mse_class_train = []
        # self.mse_class_val = []
        # self.mse_class_test = []

        # self.traitAccTrain = []
        # self.meanAvgAccAllTrain = []
        # self.traitAccVal = []
        # self.meanAvgAccAllVal = []
        
        # self.traitAccTest = []
        # self.meanAvgAccAllTest = []

        self.loss_values_train = openFilePkl('audioMean/lossValuesTrain.pickle')
        self.loss_values_val = openFilePkl('audioMean/lossValuesVal.pickle')
        #self.loss_values_test = openFilePkl('audioMean/lossValuesTest.pickle')
        self.mse_class_train = openFilePkl('audioMean/mseClassTrain.pickle')
        self.mse_class_val = openFilePkl('audioMean/mseClassVal.pickle')
        #self.mse_class_test = openFilePkl('audioMean/mseClassTest.pickle')
        self.traitAccTrain = openFilePkl('audioMean/traitAccTrain.pickle')
        self.meanAvgAccAllTrain = openFilePkl('audioMean/meanAvgAccAllTrain.pickle')
        self.traitAccVal = openFilePkl('audioMean/traitAccVal.pickle')
        self.meanAvgAccAllVal = openFilePkl('audioMean/meanAvgAccAllVal.pickle')
                
        #self.traitAccTest = openFilePkl('audioMean/traitAccTest.pickle')
        #self.meanAvgAccAllTest = openFilePkl('audioMean/meanAvgAccAllTest.pickle')
    
    def optimizer_to(self, optim, device):
        #Code optimizer_to from: https://github.com/pytorch/pytorch/issues/8741
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
    
    def repackage_hidden(self, h):
        #source: https://github.com/pytorch/examples/blob/main/word_language_model/main.py#L110
    #"""Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def train(self, train_dl, epoch):
        self.net.train()
        #self.netVggish.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        mseClassEpoch = torch.zeros(1, len(self.classes)).to(self.device)
        traitAccPerBatch = torch.zeros(1, len(self.classes)).to(self.device)

        endCheck = 0

        print('Epoch: ', epoch)

        for batch in train_dl:
            #print(batch['audio'].shape)

            #hidden = hidden.detach() 
            #detach hidden: read more https://discuss.pytorch.org/t/why-i-must-set-retain-graph-true-so-that-my-program-can-run-without-error/12428
            #hidden[0].detach_()
            #hidden[1].detach_()
            # audioBatch = batch['audio'] #.to(self.device)
            # textBatch = batch['transcription']
            # inputAudioVggish = self.netVggish(audioBatch) #.to(self.device)
            #inputTextToken = self.tokenizer(textBatch, padding="max_length", truncation=True, return_tensors="pt")
            inputAudioOneSecond, labels, lengthAudioSecond = batch['audioPerSecond'].to(self.device), batch['groundtruth'].float().to(self.device), batch['LenAudioSecond']
            #print('Input and Label Device: {}, {} '.format(inputs.device, labels.device))

            self.optim.zero_grad()

            #forward + backward + optimize
            pred = self.net(inputAudioOneSecond, lengthAudioSecond)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            #hidden[0].detach_()
            #hidden[1].detach_()

            #print('prediction: ', pred)
            #print('shape pred: ', pred.shape)

            loss = self.criterion(pred, labels) #MSE Loss
            #print('Endcheck: ', endCheck)
            #print(len(train_dl)-1)
            
            #loss.backward(retain_graph=True)
            loss.backward()

            self.optim.step() # weight update

            lossPerClass = traitsMSE(pred, labels)
            endCheck += 1
            #Average Accuracy
            traitsAcc = traitsAverageAccuracy(pred, labels)
            traitAccPerBatch = torch.add(traitAccPerBatch, traitsAcc)

            #print('Accuracy per Trait per Batch: ', traitsAcc)
    
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(pred.data, 1)
            _, labelmax = torch.max(labels, 1) #groundtruth max
            #print('Predicted: ', predicted)
            #print('Label groundtruth max: ', labelmax)
            total += labels.size(0)
            #print('Total label size: ', total)
            #print('Label size: ', labels.size(0))
            #print(len(train_dl))
            correct += (predicted == labelmax).sum().item()
            mseClassEpoch = torch.add(mseClassEpoch, lossPerClass)

            # cumulative loss
            running_loss += loss.item()
    
            #save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': loss
            }, self.pathRes+'/checkpoint/audio_parameters_checkpoint.pt')
    
        print('Average Accuracy per Trait: ', traitAccPerBatch / len(train_dl))
        self.traitAccTrain.append(traitAccPerBatch / len(train_dl))
        meanAvgAccPerEpoch = meanAverageAccuracy(traitAccPerBatch / len(train_dl), len(self.classes))
        self.meanAvgAccAllTrain.append(meanAvgAccPerEpoch.detach().cpu().numpy())
        print('Mean Average Accuracy: ', meanAvgAccPerEpoch)
        print('Epoch: {}, Loss Training: {}'.format(epoch + 1, running_loss / len(train_dl)))       
        self.loss_values_train.append(running_loss / len(train_dl))
        print('Epoch: {}, Loss per Class Training: {}'.format(epoch + 1, torch.div(mseClassEpoch, len(train_dl))))    
        self.mse_class_train.append(torch.div(mseClassEpoch, len(train_dl)))
        saveFilePkl(self.pathRes +'/lossValuesTrain.pickle', self.loss_values_train)
        saveFilePkl(self.pathRes+'/mseClassTrain.pickle', self.mse_class_train)
        saveFilePkl(self.pathRes+'/traitAccTrain.pickle', self.traitAccTrain)
        saveFilePkl(self.pathRes+'/meanAvgAccAllTrain.pickle', self.meanAvgAccAllTrain)
    
    def val(self, val_dl, epoch):
        #evaluation
        self.net.eval()
        #self.netVggish.eval()
        loss_val = 0.0
        correct = 0
        mseClassEpochVal = torch.zeros(1, len(self.classes)).to(self.device)
        traitAccPerBatch = torch.zeros(1, len(self.classes)).to(self.device)
        total = 0

    
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for batch in val_dl:
                inputAudioOneSecond, labels, lengthAudioSecond = batch['audioPerSecond'].to(self.device), batch['groundtruth'].float().to(self.device), batch['LenAudioSecond']
                #print('Input and Label Device: {}, {} '.format(inputs.device, labels.device))
                pred = self.net(inputAudioOneSecond, lengthAudioSecond)
                #loss mse
                loss = self.criterion(pred, labels)
                lossPerClass = traitsMSE(pred, labels)

                loss_val += loss.item()
                mseClassEpochVal = torch.add(mseClassEpochVal, lossPerClass)

                #print('Prediction: ', pred)
                #print('Groundtruth: ', labels)

                #Average Accuracy
                traitsAcc = traitsAverageAccuracy(pred, labels)
                traitAccPerBatch = torch.add(traitAccPerBatch, traitsAcc)

                #print('Accuracy per Trait per Batch: ', traitsAcc)
    
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(pred.data, 1)
                _, labelmax = torch.max(labels, 1) #groundtruth max
                
                #print('Predicted max: ', predicted)
                #print('Label groundtruth max: ', labelmax)
                total += labels.size(0)
                correct += (predicted == labelmax).sum().item()
    
        #scheduler for checking the improvement of val loss
        self.scheduler.step(loss_val / len(val_dl)) 
        
        print('Average Accuracy per Trait: ', traitAccPerBatch / len(val_dl))
        self.traitAccVal.append(traitAccPerBatch / len(val_dl))
        meanAvgAccPerEpoch = meanAverageAccuracy(traitAccPerBatch / len(val_dl), len(self.classes))
        self.meanAvgAccAllVal.append(meanAvgAccPerEpoch.detach().cpu().numpy())
        print('Mean Average Accuracy: ', meanAvgAccPerEpoch)
        print('Epoch: {}, Loss Val: {}'.format(epoch + 1, loss_val / len(val_dl)))       
        self.loss_values_val.append(loss_val / len(val_dl))
        print('Epoch: {}, Loss per Class Validation: {}'.format(epoch + 1, torch.div(mseClassEpochVal, len(val_dl))))    
        self.mse_class_val.append(torch.div(mseClassEpochVal, len(val_dl)))
        saveFilePkl(self.pathRes+'/lossValuesVal.pickle', self.loss_values_val)
        saveFilePkl(self.pathRes+ '/mseClassVal.pickle', self.mse_class_val)
        saveFilePkl(self.pathRes+ '/traitAccVal.pickle', self.traitAccVal)
        saveFilePkl(self.pathRes+ '/meanAvgAccAllVal.pickle', self.meanAvgAccAllVal)
    
    def test(self, test_dl, epoch):
        #evaluation
        self.net.eval()
        #self.netVggish.eval()
        loss_test = 0.0
        correct = 0
        mseClassEpochTest = torch.zeros(1, len(self.classes)).to(self.device)
        traitAccPerBatch = torch.zeros(1, len(self.classes)).to(self.device)
        total = 0
    
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for batch in test_dl:
                inputAudioOneSecond, labels, lengthAudioSecond = batch['audioPerSecond'].to(self.device), batch['groundtruth'].float().to(self.device), batch['LenAudioSecond']
                #print('Input and Label Device: {}, {} '.format(inputs.device, labels.device))
                #forward + backward + optimize
                pred = self.net(inputAudioOneSecond, lengthAudioSecond)

                #loss mse
                loss = self.criterion(pred, labels)
                lossPerClass = traitsMSE(pred, labels)

                loss_test += loss.item()
                mseClassEpochTest = torch.add(mseClassEpochTest, lossPerClass)

                #print('Prediction: ', pred)
                #pred = outputs["question_answering_score"].argmax()
                #print('Groundtruth: ', labels)
                #print("prediction from LXMERT GQA:", gqa_answers[pred_gqa])

                #Average Accuracy
                traitsAcc = traitsAverageAccuracy(pred, labels)
                traitAccPerBatch = torch.add(traitAccPerBatch, traitsAcc)

                #print('Accuracy per Trait per Batch: ', traitsAcc)
    
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(pred.data, 1)
                _, labelmax = torch.max(labels, 1) #groundtruth max
                
                #print('Predicted max: ', predicted)
                #print('Label groundtruth max: ', labelmax)
                total += labels.size(0)
                correct += (predicted == labelmax).sum().item()
    
        print('Average Accuracy per Trait: ', traitAccPerBatch / len(test_dl))
        self.traitAccTest.append(traitAccPerBatch / len(test_dl))
        meanAvgAccPerEpoch = meanAverageAccuracy(traitAccPerBatch / len(test_dl), len(self.classes))
        self.meanAvgAccAllTest.append(meanAvgAccPerEpoch.detach().cpu().numpy())
        print('Mean Average Accuracy: ', meanAvgAccPerEpoch)
        print('Epoch: {}, Loss Test: {}'.format(epoch + 1, loss_test / len(test_dl)))       
        self.loss_values_test.append(loss_test / len(test_dl))
        print('Epoch: {}, Loss per Class Test: {}'.format(epoch + 1, torch.div(mseClassEpochTest, len(test_dl))))    
        self.mse_class_test.append(torch.div(mseClassEpochTest, len(test_dl)))
        saveFilePkl(self.pathRes+'/lossValuesTest.pickle', self.loss_values_test)
        saveFilePkl(self.pathRes+ '/mseClassTest.pickle', self.mse_class_test)
        saveFilePkl(self.pathRes+ '/traitAccTest.pickle', self.traitAccTest)
        saveFilePkl(self.pathRes+ '/meanAvgAccAllTest.pickle', self.meanAvgAccAllTest)
    
    def saveModelResults(self, delta_time=''):
        #save model
        PATH = self.pathRes+ '/parameters_finetuning.pth'
        torch.save(self.net.state_dict(), PATH)
        
        print('Train Loss: ', self.loss_values_train[-1])
        print('Val Loss: ', self.loss_values_val[-1])
        print('Train Loss per Trait: ', self.mse_class_train[-1])
        print('Val Loss per Trait: ', self.mse_class_val[-1])
        print('Train Mean Average Accuracy: ', self.meanAvgAccAllTrain[-1])
        print('Val Mean Average Accuracy: ', self.meanAvgAccAllVal[-1])
        print('Train Average Accuracy per Trait: ', self.traitAccTrain[-1])
        print('Val Average Accuracy per Trait: ', self.traitAccVal[-1])
        
        #save the results
        # res_dic = {
        #     'Prediction Time' : delta_time, 
        #     'Train Loss' : self.loss_values_train[-1], 
        #     'Val Loss' : self.loss_values_val[-1], 
        #     'Test Loss' : self.loss_values_test[-1], 
        #     'Train Loss per Trait:' : self.mse_class_train[-1], 
        #     'Val Loss per Trait:' : self.mse_class_val[-1], 
        #     'Test Loss per Trait:' : self.mse_class_test[-1],
        #     'Train Mean Average Accuracy' : self.meanAvgAccAllTrain[-1], 
        #     'Val Mean Average Accuracy' : self.meanAvgAccAllVal[-1], 
        #     'Test Mean Average Accuracy' : self.meanAvgAccAllTest[-1], 
        #     'Train Average Accuracy per Trait' :  self.traitAccTrain[-1], 
        #     'Val Average Accuracy per Trait' :  self.traitAccVal[-1],
        #     'Test Average Accuracy per Trait' :  self.traitAccTest[-1]
        #     }
        #save the results
        res_dic = {
            'Prediction Time' : delta_time, 
            'Train Loss' : self.loss_values_train[-1], 
            'Val Loss' : self.loss_values_val[-1], 
            'Train Loss per Trait:' : self.mse_class_train[-1], 
            'Val Loss per Trait:' : self.mse_class_val[-1], 
            'Train Mean Average Accuracy' : self.meanAvgAccAllTrain[-1], 
            'Val Mean Average Accuracy' : self.meanAvgAccAllVal[-1], 
            'Train Average Accuracy per Trait' :  self.traitAccTrain[-1], 
            'Val Average Accuracy per Trait' :  self.traitAccVal[-1],
            }

        saveResults(res_dic, self.pathRes+ "/results_trainValTest.txt")

        #Plot Loss
        #plotResults(self.loss_values_train, self.loss_values_val, self.loss_values_test, len(self.loss_values_train))
        plotResults(self.loss_values_train, self.loss_values_val, len(self.loss_values_train))
        #Plot Accuracy
        #plotResults(self.meanAvgAccAllTrain, self.meanAvgAccAllVal, self.meanAvgAccAllTest, len(self.meanAvgAccAllTrain), loss=False)
        plotResults(self.meanAvgAccAllTrain, self.meanAvgAccAllVal, len(self.meanAvgAccAllTrain), loss=False)