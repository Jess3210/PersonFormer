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

from transformers import BertConfig, BertModel, BertTokenizer

from architectures.visionClassTrainingScratch import visionConcat as visionClass
from architectures.textFinetuning import textFinetuning as textClass
from architectures.audioClass_1s import audioConcat as audioClass

from scripts_packages.resultsFileManagement import * #load/save pickle, plot results
from scripts_packages.metricLoss import * #mean average accuracy, average accuracy, mse per trait

class visionAudioTextFusion(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        faceCheckpoint = './checkpoint/visionFullTrainingLR0.01/parameters_finetuning.pth', #trained weights if available
        personCheckpoint = './checkpoint/visionFullTrainingPersonLR0.001/parameters_finetuning.pth', #trained weights if available
        sceneCheckpoint = './checkpoint/visionFullTrainingScene/parameters_finetuning.pth', #trained weights if available
        textCheckpoint = './checkpoint/text_parameters_finetuning.pth', #trained weights if available
        audioCheckpoint = './checkpoint/audio_1s_parameters_finetuning.pth', #trained weights if available
        input_vision=768,
        output_vision=512,
        dropout_p = 0.2,
        ):
        
        super(visionAudioTextFusion, self).__init__()

        #load face model checkpoint
        self.vision_module_scene = visionClass()
        self.vision_module_scene.load_state_dict(torch.load(sceneCheckpoint))
        
        #load face model checkpoint
        self.vision_module_face = visionClass()
        self.vision_module_face.load_state_dict(torch.load(faceCheckpoint))

        #load person model checkpoint
        self.vision_module_person = visionClass()
        self.vision_module_person.load_state_dict(torch.load(personCheckpoint))

        self.audio_module = audioClass()
        self.audio_module.load_state_dict(torch.load(audioCheckpoint))

        self.text_module = textClass()
        self.text_module.load_state_dict(torch.load(textCheckpoint))
        
        #Freeze early layers
        for param in self.vision_module_face.parameters():
            param.requires_grad = False

        for param in self.vision_module_person.parameters():
            param.requires_grad = False
        
        for param in self.vision_module_scene.parameters():
            param.requires_grad = False
        
        for param in self.audio_module.parameters():
            param.requires_grad = False

        self.vision_module_face.fc2 = nn.Linear(input_vision, output_vision, bias=True)
        self.vision_module_person.fc2 = nn.Linear(input_vision, output_vision, bias=True)
        self.vision_module_scene.fc2 = nn.Linear(input_vision, output_vision, bias=True)

        self.audio_module.fc = nn.Linear(128, 128, bias=True)
        self.text_module.fc = nn.Linear(768, 768, bias=True)

        self.fusionFacePerson = torch.nn.Linear(
            in_features=(output_vision+output_vision), 
            out_features=(output_vision+output_vision)
        )

        #get fusion layer from 1024 to 512
        self.FacePersonlayer = torch.nn.Linear( 
            in_features=(output_vision+output_vision), 
            out_features=(output_vision)
        )

        self.fusion_avt = torch.nn.Linear(
            in_features=(768+128+512), 
            out_features=1024
        )


        self.fusion_avtfp_hidden = torch.nn.Linear( 
            in_features=(1024+512), 
            out_features=1024
        )
        
        self.output = torch.nn.Linear(
            in_features=1024, 
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, imageFace, imagePerson, imageScene, audio, text): 
        visionFace = self.vision_module_face(imageFace)
        visionPerson = self.vision_module_person(imagePerson)
        visionScene = self.vision_module_scene(imageScene)
        textfeat = self.text_module(text)
        audiofeat = self.audio_module(audio)

        #projection layer person + face
        combinedPersonFace = torch.cat(
            [visionFace, visionPerson], dim=1
        )

        #projection layer scene + audio + text
        combinedVAT = torch.cat(
            [visionScene, audiofeat, textfeat], dim=1
        )
        #hidden layer scene + audio + text
        fusedAVT = self.dropout(
            torch.nn.functional.relu(
                self.fusion_avt(combinedVAT)
            )
        )
        #hidden layer face + person
        fusedFacePerson = self.dropout(
            torch.nn.functional.relu(
            self.fusionFacePerson(combinedPersonFace)
            )
        )
        #hidden layer face + person
        hiddenLayerFacePerson = self.dropout(
            torch.nn.functional.relu(
                self.FacePersonlayer(fusedFacePerson)
            )
        )
        #projection layer scene + audio + text
        combinedVATFP = torch.cat(
            [fusedAVT, hiddenLayerFacePerson], dim=1
        )
        hiddenLayerVATFP = self.dropout(
            torch.nn.functional.relu(
                self.fusion_avtfp_hidden(combinedVATFP)
            )
        )
        logits = self.output(hiddenLayerVATFP)
        pred = torch.nn.functional.sigmoid(logits)
        
        return pred

class visionAudioTextFusionClass():
    def __init__(self, optimizer='sgd', lossfnct='mse', model='inst', learningrate=1e-5, device='cuda'):

        self.pathRes = './visionAudioTextM1'
        isExist = os.path.exists(self.pathRes)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.pathRes)
            os.makedirs(self.pathRes + '/checkpoint')

        self.net = visionAudioTextFusion()
       
        if(optimizer == 'sgd'):
            self.optim = torch.optim.SGD(self.net.parameters(), lr=learningrate, momentum=0.9)
        # else:
        #self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.02)

        self.criterion = nn.MSELoss()
        self.classes = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        # checkpointLoader = torch.load('visionAudioTextM1/checkpoint/visionAudioTextPersonFaceFusion_parameters_checkpoint.pt')
        # self.net.load_state_dict(checkpointLoader['model_state_dict'])
        # self.optim.load_state_dict(checkpointLoader['optimizer_state_dict'])

        #send optimizer to GPU
        self.optimizer_to(self.optim, self.device)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5, factor=0.1)
        
        self.net.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #self.netVggish.to('cpu')
        #self.netVggish.eval()

        #print('Network device: ', next(self.net.parameters()).device)
        
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

        # self.loss_values_train = openFilePkl('visionAudioTextM1/lossValuesTrain.pickle')
        # self.loss_values_val = openFilePkl('visionAudioTextM1/lossValuesVal.pickle')
        # #self.loss_values_test = openFilePkl('visionAudioTextM1/lossValuesTest.pickle')
        # self.mse_class_train = openFilePkl('visionAudioTextM1/mseClassTrain.pickle')
        # self.mse_class_val = openFilePkl('visionAudioTextM1/mseClassVal.pickle')
        # #self.mse_class_test = openFilePkl('visionAudioTextM1/mseClassTest.pickle')
        # self.traitAccTrain = openFilePkl('visionAudioTextM1/traitAccTrain.pickle')
        # self.meanAvgAccAllTrain = openFilePkl('visionAudioTextM1/meanAvgAccAllTrain.pickle')
        # self.traitAccVal = openFilePkl('visionAudioTextM1/traitAccVal.pickle')
        # self.meanAvgAccAllVal = openFilePkl('visionAudioTextM1/meanAvgAccAllVal.pickle')
                
        # # self.traitAccTest = openFilePkl('visionAudioTextM1/traitAccTest.pickle')
        # # self.meanAvgAccAllTest = openFilePkl('visionAudioTextM1/meanAvgAccAllTest.pickle')
    
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
            textBatch = batch['transcription']
            inputTextToken = self.tokenizer(textBatch, padding="max_length", truncation=True, return_tensors="pt")
            inputImageFace, inputImagePerson, inputImageScene, audio, text, labels, name = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['imageScene'].to(self.device), batch['audioPerSecond'].to(self.device), inputTextToken.to(self.device), batch['groundtruth'].float().to(self.device), batch['name']
            self.optim.zero_grad()
    
            #forward + backward + optimize
            pred = self.net(inputImageFace, inputImagePerson, inputImageScene, audio, text)
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
            }, self.pathRes+'/checkpoint/visionAudioTextPersonFaceFusion_parameters_checkpoint.pt')
    
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
                textBatch = batch['transcription']
                inputTextToken = self.tokenizer(textBatch, padding="max_length", truncation=True, return_tensors="pt")
                inputImageFace, inputImagePerson, inputImageScene, audio, text, labels, name = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['imageScene'].to(self.device), batch['audioPerSecond'].to(self.device), inputTextToken.to(self.device), batch['groundtruth'].float().to(self.device), batch['name']
                pred = self.net(inputImageFace, inputImagePerson, inputImageScene, audio, text)
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
                textBatch = batch['transcription']
                inputTextToken = self.tokenizer(textBatch, padding="max_length", truncation=True, return_tensors="pt")
                inputImageFace, inputImagePerson, inputImageScene, audio, text, labels, name = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['imageScene'].to(self.device), batch['audioPerSecond'].to(self.device), inputTextToken.to(self.device), batch['groundtruth'].float().to(self.device), batch['name']
                #forward + backward + optimize
                pred = self.net(inputImageFace, inputImagePerson, inputImageScene, audio, text)
                #loss mse
                loss = self.criterion(pred, labels)
                lossPerClass = traitsMSE(pred, labels)

                loss_test += loss.item()
                mseClassEpochTest = torch.add(mseClassEpochTest, lossPerClass)

                #Average Accuracy
                traitsAcc = traitsAverageAccuracy(pred, labels)
                traitAccPerBatch = torch.add(traitAccPerBatch, traitsAcc)

                #print('Accuracy per Trait per Batch: ', traitsAcc)
    
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
