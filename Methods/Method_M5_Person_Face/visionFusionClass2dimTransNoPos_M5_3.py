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
from visionClassTrainingScratch import visionConcat as visionClass

class visionFusion(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        faceCheckpoint = './visionFullTrainingLR0.01/parameters_finetuning.pth',
        personCheckpoint = './visionFullTrainingPersonLR0.001/parameters_finetuning.pth',
        input_vision=768,
        output_vision=512,
        d_model=512,
        nhead = 8,
        dimFeedforward = 2048,
        numLayers = 2,
        dropoutTransformer=0.1,
        dropout_p = 0.2,
        ):
        
        super(visionFusion, self).__init__()
        
        #load face model checkpoint
        self.vision_module_face = visionClass()
        self.vision_module_face.load_state_dict(torch.load(faceCheckpoint))

        #load person model checkpoint
        self.vision_module_person = visionClass()
        self.vision_module_person.load_state_dict(torch.load(personCheckpoint))
        
        #Freeze early layers
        for param in self.vision_module_face.parameters():
            param.requires_grad = False

        for param in self.vision_module_person.parameters():
            param.requires_grad = False

        self.vision_module_face.fc2 = nn.Linear(input_vision, output_vision, bias=True)
        self.vision_module_person.fc2 = nn.Linear(input_vision, output_vision, bias=True)

        #self.pos_encoder_personFace = PositionalEncoding(d_model=d_model, dropout=dropoutTransformer, max_len=2)
        #layer1
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer1_1 = nn.Linear(d_model, dimFeedforward)
        self.fctransformer1_2 = nn.Linear(dimFeedforward, d_model)

        #layer2
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer2_1 = nn.Linear(d_model, dimFeedforward)
        self.fctransformer2_2 = nn.Linear(dimFeedforward, d_model)

        self.fusionFacePerson = nn.Linear(output_vision+output_vision, output_vision)
        
        self.fc = torch.nn.Linear(
            in_features=(output_vision), 
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)
        self.dropoutTransf = torch.nn.Dropout(dropoutTransformer)
        
    def forward(self, imageFace, imagePerson): 
        visionFace = self.vision_module_face(imageFace)
        visionPerson = self.vision_module_person(imagePerson)
        fused = torch.stack(
            [visionFace, visionPerson], dim=0
        )
        residual = fused

        facePerson_features = self.self_attn(fused, fused, fused)[0]
        facePerson_features += residual
        facePerson_features = self.layer_norm(facePerson_features)
        residual = facePerson_features
        facePerson_features = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer1_1(facePerson_features)))
        facePerson_features = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer1_2(facePerson_features)))
        facePerson_features += residual
        facePerson_features = self.layer_norm(facePerson_features)

        #layer2 Transformer
        residual = facePerson_features
        facePerson_features = self.self_attn2(facePerson_features, facePerson_features, facePerson_features)[0]
        facePerson_features += residual
        facePerson_features = self.layer_norm(facePerson_features)
        residual = facePerson_features
        facePerson_features = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer2_1(facePerson_features)))
        facePerson_features = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer2_2(facePerson_features)))
        facePerson_features += residual
        facePerson_features = self.layer_norm(facePerson_features)

        combinedOutput = torch.reshape(facePerson_features, (-1, 1024))

        outputDim = self.dropout(
            torch.nn.functional.relu(
            self.fusionFacePerson(combinedOutput)
            )
        )

        logits = self.fc(outputDim)
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
        if embedding.dim() == 3:
            embedding = embedding + self.pe[:embedding.size(0)]
            drop = self.dropout(embedding)
            #from seqLen, BS, embedding to BS, seqLen, embedding
            outputEmbedding = torch.permute(drop, (1, 0, 2))
        else:
            embedding = torch.unsqueeze(embedding, dim=-1)
            #(512, BS, 1)
            embedding = torch.permute(embedding, (1, 0, 2))
            embedding = embedding + self.pe[:embedding.size(0)]
            drop = self.dropout(embedding)
            outputEmbedding = torch.permute(drop, (1, 0, 2))
            outputEmbedding = torch.squeeze(outputEmbedding)
        return(outputEmbedding)

class visionFusionClass():
    def __init__(self, optimizer='sgd', lossfnct='mse', model='inst', learningrate=1e-5, device='cuda'):

        self.pathRes = './visionFacePersonFusionTransformer'
        isExist = os.path.exists(self.pathRes)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.pathRes)
            os.makedirs(self.pathRes + '/checkpoint')

        self.net = visionFusion()
       
        if(optimizer == 'sgd'):
            self.optim = torch.optim.SGD(self.net.parameters(), lr=learningrate, momentum=0.9)
        # else:
        #self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.02)

        self.criterion = nn.MSELoss()
        self.classes = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        # checkpointLoader = torch.load('visionFacePersonFusionTransformer/checkpoint/visionFusionFacePerson_parameters_checkpoint.pt')
        # self.net.load_state_dict(checkpointLoader['model_state_dict'])
        # self.optim.load_state_dict(checkpointLoader['optimizer_state_dict'])

        #send optimizer to GPU
        #self.optimizer_to(self.optim, self.device)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5, factor=0.1)
        
        self.net.to(self.device)
        
        self.loss_values_train = []
        self.loss_values_val = []
        self.loss_values_test = []
        
        self.mse_class_train = []
        self.mse_class_val = []
        self.mse_class_test = []

        self.traitAccTrain = []
        self.meanAvgAccAllTrain = []
        self.traitAccVal = []
        self.meanAvgAccAllVal = []
        
        self.traitAccTest = []
        self.meanAvgAccAllTest = []

        # self.loss_values_train = openFilePkl('visionFacePersonFusionTransformer/lossValuesTrain.pickle')
        # self.loss_values_val = openFilePkl('visionFacePersonFusionTransformer/lossValuesVal.pickle')
        # #self.loss_values_test = openFilePkl('visionSwinFinetuning/lossValuesTest.pickle')
        # self.mse_class_train = openFilePkl('visionFacePersonFusionTransformer/mseClassTrain.pickle')
        # self.mse_class_val = openFilePkl('visionFacePersonFusionTransformer/mseClassVal.pickle')
        # #self.mse_class_test = openFilePkl('visionSwinFinetuning/mseClassTest.pickle')
        # self.traitAccTrain = openFilePkl('visionFacePersonFusionTransformer/traitAccTrain.pickle')
        # self.meanAvgAccAllTrain = openFilePkl('visionFacePersonFusionTransformer/meanAvgAccAllTrain.pickle')
        # self.traitAccVal = openFilePkl('visionFacePersonFusionTransformer/traitAccVal.pickle')
        # self.meanAvgAccAllVal = openFilePkl('visionFacePersonFusionTransformer/meanAvgAccAllVal.pickle')
                
        # # self.traitAccTest = openFilePkl('visionSwinFinetuning/traitAccTest.pickle')
        # # self.meanAvgAccAllTest = openFilePkl('visionSwinFinetuning/meanAvgAccAllTest.pickle')
    
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
        running_loss = 0.0
        correct = 0
        total = 0
        mseClassEpoch = torch.zeros(1, len(self.classes)).to(self.device)
        traitAccPerBatch = torch.zeros(1, len(self.classes)).to(self.device)

        print('Epoch: ', epoch)
        for batch in train_dl:
            inputImageFace, inputImagePerson, labels = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['groundtruth'].float().to(self.device)
            self.optim.zero_grad()
    
            #forward + backward + optimize
            pred = self.net(inputImageFace, inputImagePerson)
            loss = self.criterion(pred, labels) #MSE Loss
            loss.backward() # backward pass
            self.optim.step() # weight update

            lossPerClass = traitsMSE(pred, labels)

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
            }, self.pathRes+'/checkpoint/visionFusionFacePerson_parameters_checkpoint.pt')
    
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
        loss_val = 0.0
        correct = 0
        mseClassEpochVal = torch.zeros(1, len(self.classes)).to(self.device)
        traitAccPerBatch = torch.zeros(1, len(self.classes)).to(self.device)
        total = 0
    
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for batch in val_dl:
                inputImageFace, inputImagePerson, labels = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['groundtruth'].float().to(self.device)
                pred = self.net(inputImageFace, inputImagePerson)
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
        loss_test = 0.0
        correct = 0
        mseClassEpochTest = torch.zeros(1, len(self.classes)).to(self.device)
        traitAccPerBatch = torch.zeros(1, len(self.classes)).to(self.device)
        total = 0
    
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for batch in test_dl:
                inputImageFace, inputImagePerson, labels = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['groundtruth'].float().to(self.device)
                #forward + backward + optimize
                pred = self.net(inputImageFace, inputImagePerson)
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
        #print('Test Loss: ', self.loss_values_test[-1])
        print('Train Loss per Trait: ', self.mse_class_train[-1])
        print('Val Loss per Trait: ', self.mse_class_val[-1])
        #print('Test Loss per Trait: ', self.mse_class_test[-1])
        print('Train Mean Average Accuracy: ', self.meanAvgAccAllTrain[-1])
        print('Val Mean Average Accuracy: ', self.meanAvgAccAllVal[-1])
        #print('Test Mean Average Accuracy: ', self.meanAvgAccAllTest[-1])
        print('Train Average Accuracy per Trait: ', self.traitAccTrain[-1])
        print('Val Average Accuracy per Trait: ', self.traitAccVal[-1])
        #print('Test Average Accuracy per Trait: ', self.traitAccTest[-1])
        
        #save the results
        res_dic = {
            'Prediction Time' : delta_time, 
            'Train Loss' : self.loss_values_train[-1], 
            'Val Loss' : self.loss_values_val[-1], 
            #'Test Loss' : self.loss_values_test[-1], 
            'Train Loss per Trait:' : self.mse_class_train[-1], 
            'Val Loss per Trait:' : self.mse_class_val[-1], 
            #'Test Loss per Trait:' : self.mse_class_test[-1],
            'Train Mean Average Accuracy' : self.meanAvgAccAllTrain[-1], 
            'Val Mean Average Accuracy' : self.meanAvgAccAllVal[-1], 
            #'Test Mean Average Accuracy' : self.meanAvgAccAllTest[-1], 
            'Train Average Accuracy per Trait' :  self.traitAccTrain[-1], 
            'Val Average Accuracy per Trait' :  self.traitAccVal[-1],
            #'Test Average Accuracy per Trait' :  self.traitAccTest[-1]
            }

        saveResults(res_dic, self.pathRes+ "/results_trainValTest.txt")

        #Plot Loss
        plotResults(self.loss_values_train, self.loss_values_val, len(self.loss_values_train), pathSave=self.pathRes)
        #Plot Accuracy
        plotResults(self.meanAvgAccAllTrain, self.meanAvgAccAllVal, len(self.meanAvgAccAllTrain), loss=False, pathSave=self.pathRes)
