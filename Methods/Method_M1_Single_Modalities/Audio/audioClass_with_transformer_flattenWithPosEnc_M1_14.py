import torch
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
import os
from torchvggish import vggish, vggish_input
import math

from scripts_packages.resultsFileManagement import * #load/save pickle, plot results
from scripts_packages.metricLoss import * #mean average accuracy, average accuracy, mse per trait

#from architectures.vggish import VGGish
from architectures.vggish2 import VGGish
from torchvggish import vggish_input

class audioConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        #audio_model = VGGish(None), #load pretrained weights AudioSet: VGGish(torch.load('./checkpoint/pytorch_vggish.pth'))
        audio_model = VGGish(),
        audio_output=128,
        d_model=128,
        nhead = 8,
        dimFeedforward1 = 1024,
        numLayers = 2,
        num_layers_lstm = 1,
        dropout_p = 0.2,
        dropoutTransformer=0.1
        ):
        super(audioConcat, self).__init__()

        self.audio_module = audio_model.to(torch.device('cuda:0'))
        
        self.audio_module = self.audio_module.to(torch.device('cuda:0'))
        #self.audio_module.load_state_dict(torch.load('pytorch_vggish.pth')) load pretrained weights 
        
        self.pos_encoder_audio = PositionalEncoding(d_model=d_model, dropout=dropoutTransformer)

        #layer1
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer1_1 = nn.Linear(d_model, dimFeedforward1)
        self.fctransformer1_2 = nn.Linear(dimFeedforward1, d_model)

        #layer2
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer2_1 = nn.Linear(d_model, dimFeedforward1)
        self.fctransformer2_2 = nn.Linear(dimFeedforward1, d_model)

        self.flatten = nn.Flatten() #flatten from (BS, 15, 128) to (BS, 1920)

        self.fc = torch.nn.Linear(
            in_features=1920, 
            out_features=d_model
        )

        self.fc2 = torch.nn.Linear(
            in_features=d_model, 
            out_features=num_classes
        )
        self.dropoutTransf = torch.nn.Dropout(dropoutTransformer)
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, audio, audiolen): #please give us input already embedded vggish
        audio_features = torch.empty((len(audio), 15, 128)).to(torch.device('cuda:0'))
        audiofeat = audio.reshape(-1, 1, 96, 64)
        vggEmb = self.audio_module(audiofeat)
        vggEmb = vggEmb.reshape(-1, 15, vggEmb.shape[-1])

        vggEmb = self.pos_encoder_audio(vggEmb)
        
        residual = vggEmb
        audio_features = self.self_attn1(vggEmb, vggEmb, vggEmb)[0]
        audio_features += residual
        audio_features = self.layer_norm1(audio_features)
        residual = audio_features
        audio_features = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer1_1(audio_features)))
        audio_features = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer1_2(audio_features)))
        audio_features += residual
        audio_features = self.layer_norm1(audio_features)

        #layer2 Transformer
        residual = audio_features
        audio_features = self.self_attn2(audio_features, audio_features, audio_features)[0]
        audio_features += residual
        audio_features = self.layer_norm1(audio_features)
        residual = audio_features
        audio_features = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer2_1(audio_features)))
        audio_features = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer2_2(audio_features)))
        audio_features += residual
        audio_features = self.layer_norm1(audio_features)

        audio_features = self.flatten(audio_features)
        
        fc_layer1 = self.dropout(
            torch.nn.functional.relu(self.fc(audio_features)))

        logits = self.fc2(fc_layer1)
        pred = torch.nn.functional.sigmoid(logits)
        
        return pred
    
    #code for class PositionalEncoding: based on pytorch tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#changed input Tensor shape from [batch_size, seq_len, embedding_dim] to [seq_len, batch_size, embedding_dim]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model = 128, dropout =  0.1, max_len = 15):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, embedding):
        #print('Embedding shape: ', embedding.shape)
        if embedding.dim() == 3:
            embedding = torch.permute(embedding, (1, 0, 2))
            #print('Embedding permute: ', embedding.shape)
            embedding = embedding + self.pe[:embedding.size(0)]
            drop = self.dropout(embedding)
            outputEmbedding = torch.permute(drop, (1, 0, 2))
        else:
            embedding = torch.unsqueeze(embedding, dim=-1)
            #(512, BS, 1)
            embedding = torch.permute(embedding, (1, 0, 2))
            #print('Embedding permute: ', embedding.shape)
            embedding = embedding + self.pe[:embedding.size(0)]
            drop = self.dropout(embedding)
            outputEmbedding = torch.permute(drop, (1, 0, 2))
            #print('Embedding permute: ', outputEmbedding.shape)
            outputEmbedding = torch.squeeze(outputEmbedding)
            #print('Embedding back squeeze: ', outputEmbedding.shape)
        return(outputEmbedding)

class audioClass():
    def __init__(self, optimizer='sgd', lossfnct='mse', learningrate=1e-5, device='cuda', bs=32):

        self.pathRes = './audioMean'
        isExist = os.path.exists(self.pathRes)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.pathRes)
            os.makedirs(self.pathRes + '/checkpoint')

        self.net = audioConcat()

        self.batch_size = bs
       
        if(optimizer == 'sgd'):
            self.optim = torch.optim.SGD(self.net.parameters(), lr=learningrate, momentum=0.9)
        
        self.criterion = nn.MSELoss()
        self.classes = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        #---load checkpoint during training stop ----
        # checkpointLoader = torch.load('audioMean/checkpoint/audio_parameters_checkpoint.pt')
        # self.net.load_state_dict(checkpointLoader['model_state_dict'])
        # self.optim.load_state_dict(checkpointLoader['optimizer_state_dict'])

        #send optimizer to GPU
        #self.optimizer_to(self.optim, self.device)
        
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5, factor=0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5, factor=0.1)
        
        self.net.to(self.device)
        #self.hidden = self.net.init_hidden(self.batch_size)

        #self.netVggish.to('cpu')
        #self.netVggish.eval()

        #print('Network device: ', next(self.net.parameters()).device)
        
        self.loss_values_train = []
        self.loss_values_val = []
        #self.loss_values_test = []
        
        self.mse_class_train = []
        self.mse_class_val = []
        #self.mse_class_test = []

        self.traitAccTrain = []
        self.meanAvgAccAllTrain = []
        self.traitAccVal = []
        self.meanAvgAccAllVal = []
        
        # self.traitAccTest = []
        # self.meanAvgAccAllTest = []

        #----load saved values during training stop and restart ----
        # self.loss_values_train = openFilePkl('audioMean/lossValuesTrain.pickle')
        # self.loss_values_val = openFilePkl('audioMean/lossValuesVal.pickle')
        # #self.loss_values_test = openFilePkl('audioMean/lossValuesTest.pickle')
        # self.mse_class_train = openFilePkl('audioMean/mseClassTrain.pickle')
        # self.mse_class_val = openFilePkl('audioMean/mseClassVal.pickle')
        # #self.mse_class_test = openFilePkl('audioMean/mseClassTest.pickle')
        # self.traitAccTrain = openFilePkl('audioMean/traitAccTrain.pickle')
        # self.meanAvgAccAllTrain = openFilePkl('audioMean/meanAvgAccAllTrain.pickle')
        # self.traitAccVal = openFilePkl('audioMean/traitAccVal.pickle')
        # self.meanAvgAccAllVal = openFilePkl('audioMean/meanAvgAccAllVal.pickle')
                
        # #self.traitAccTest = openFilePkl('audioMean/traitAccTest.pickle')
        # #self.meanAvgAccAllTest = openFilePkl('audioMean/meanAvgAccAllTest.pickle')
    
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
            inputAudioOneSecond, labels, lengthAudioSecond = batch['audioPerSecond'].to(self.device), batch['groundtruth'].float().to(self.device), batch['LenAudioSecond']

            self.optim.zero_grad()

            #forward + backward + optimize
            pred = self.net(inputAudioOneSecond, lengthAudioSecond)

            loss = self.criterion(pred, labels) #MSE Loss
            loss.backward()

            self.optim.step() # weight update

            lossPerClass = traitsMSE(pred, labels)
            endCheck += 1
            #Average Accuracy
            traitsAcc = traitsAverageAccuracy(pred, labels)
            traitAccPerBatch = torch.add(traitAccPerBatch, traitsAcc)
    
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(pred.data, 1)
            _, labelmax = torch.max(labels, 1) #groundtruth max
            total += labels.size(0)
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
                pred = self.net(inputAudioOneSecond, lengthAudioSecond)
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

                #Average Accuracy
                traitsAcc = traitsAverageAccuracy(pred, labels)
                traitAccPerBatch = torch.add(traitAccPerBatch, traitsAcc)
    
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(pred.data, 1)
                _, labelmax = torch.max(labels, 1) #groundtruth max
                
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
        plotResults(self.loss_values_train, self.loss_values_val, len(self.loss_values_train), pathSave=self.pathRes)
        #Plot Accuracy
        plotResults(self.meanAvgAccAllTrain, self.meanAvgAccAllVal, len(self.meanAvgAccAllTrain), loss=False, pathSave=self.pathRes)