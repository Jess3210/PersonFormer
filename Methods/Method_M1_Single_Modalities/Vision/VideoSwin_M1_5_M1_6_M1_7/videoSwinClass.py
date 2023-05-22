import torch
from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim 
import pickle
import os

from scripts_packages.resultsFileManagement import * #load/save pickle, plot results
from scripts_packages.metricLoss import * #mean average accuracy, average accuracy, mse per trait

#video swin transformer
from architectures.VideoSwinTransformer.mmcv import Config, DictAction
from architectures.VideoSwinTransformer.mmaction.models import build_model
from architectures.VideoSwinTransformer.mmcv.runner import get_dist_info, init_dist, load_checkpoint

from scripts_packages.scheduler import CyclicCosineDecayLR

class visionVideoSwin(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        #------decide which VideoSwin architecture to use and choose which checkpoint to load if necessary
        config = './architectures/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py',
        # checkpoint = './architectures/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth',
        #config = './architectures/Video-Swin-Transformer/configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py',
        #checkpoint = './architectures/checkpoints/swin_small_patch244_window877_kinetics400_1k.pth',
        # config = './architectures/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py',
        # checkpoint = './architectures/checkpoints/swin_base_patch244_window877_kinetics400_1k.pth',
        #config = './architectures/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py',
        #checkpoint = './architectures/checkpoints/swin_base_patch244_window877_kinetics400_22k.pth',
        #config = './architectures/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py',
        #checkpoint = './architectures/checkpoints/swin_base_patch244_window877_kinetics600_22k.pth',
        output_vision=768,
        dropout_p = 0.5,
        ):
        
        super(visionVideoSwin, self).__init__()
        
        #initialize Video Swin Transformer
        cfg = Config.fromfile(config)
        modelSwin = build_model(cfg.model, train_cfg=None, test_cfg=None)
        #---load checkpoint ----
        #load_checkpoint(modelSwin, checkpoint, map_location='cuda:0')
        #load_checkpoint(modelSwin, checkpoint, map_location='cpu')

        #get backone of video swin transformer
        self.vision_module = modelSwin.backbone

        # Freeze early layers
        # for param in self.vision_module.parameters():
        #     param.requires_grad = False
        
        #set to inference mode
        #self.vision_module.eval()

        self.vision_feat = nn.Linear(output_vision, output_vision, bias=True)
        self.pooling = nn.AdaptiveAvgPool3d((1,1,1))
        
        self.fc = torch.nn.Linear(
            in_features=output_vision, 
            out_features=output_vision
        )
        self.fc2 = torch.nn.Linear(
            in_features=output_vision, 
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, image): 
        vision = self.vision_module(image)
        vision = torch.squeeze(self.pooling(vision))
        image_features = self.dropout(torch.nn.functional.relu(
            self.vision_feat(vision)))
        output = self.dropout(torch.nn.functional.relu(self.fc(image_features)))
        logits = self.fc2(output)
        pred = torch.nn.functional.sigmoid(logits)
        
        return pred

class visionClass():
    def __init__(self, optimizer='sgd', lossfnct='mse', learningrate=1e-5, device='cuda'):

        self.pathRes = './videoSwinTraining'
        isExist = os.path.exists(self.pathRes)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.pathRes)
            os.makedirs(self.pathRes + '/checkpoint')

        self.net = visionVideoSwin()
       
        if(optimizer == 'sgd'):
            self.optim = torch.optim.SGD(self.net.parameters(), lr=learningrate, momentum=0.9)
            
        #for Adam optimizer
        else:
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.02)

        self.criterion = nn.MSELoss()
        self.classes = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        # checkpointLoader = torch.load('visionFullTrainingPerson/checkpoint/visionTinyScratch_parameters_checkpoint.pt')
        # self.net.load_state_dict(checkpointLoader['model_state_dict'])
        # self.optim.load_state_dict(checkpointLoader['optimizer_state_dict'])

        #send optimizer to GPU
        #self.optimizer_to(self.optim, self.device)
        
        if(optimizer == 'sgd'):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5, factor=0.1)
        #Adam optimizer
        else:
            self.scheduler = CyclicCosineDecayLR(self.optim, 
                                    init_decay_epochs=30,
                                    min_decay_lr=0.000001,
                                    warmup_epochs=3,
                                    warmup_start_lr=0.0001)
        
        self.net.to(self.device)
        
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
        
        #self.traitAccTest = []
        #self.meanAvgAccAllTest = []

        #-----load checkpoint after a training break----
        # self.loss_values_train = openFilePkl('videoSwinTraining/lossValuesTrain.pickle')
        # self.loss_values_val = openFilePkl('videoSwinTraining/lossValuesVal.pickle')
        # #self.loss_values_test = openFilePkl('videoSwinTraining/lossValuesTest.pickle')
        # self.mse_class_train = openFilePkl('videoSwinTraining/mseClassTrain.pickle')
        # self.mse_class_val = openFilePkl('videoSwinTraining/mseClassVal.pickle')
        # #self.mse_class_test = openFilePkl('videoSwinTraining/mseClassTest.pickle')
        # self.traitAccTrain = openFilePkl('videoSwinTraining/traitAccTrain.pickle')
        # self.meanAvgAccAllTrain = openFilePkl('videoSwinTraining/meanAvgAccAllTrain.pickle')
        # self.traitAccVal = openFilePkl('videoSwinTraining/traitAccVal.pickle')
        # self.meanAvgAccAllVal = openFilePkl('videoSwinTraining/meanAvgAccAllVal.pickle')
                
        # # self.traitAccTest = openFilePkl('videoSwinTraining/traitAccTest.pickle')
        # # self.meanAvgAccAllTest = openFilePkl('videoSwinTraining/meanAvgAccAllTest.pickle')
    
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
            inputImage, labels = batch['image'].to(self.device), batch['groundtruth'].float().to(self.device)
            self.optim.zero_grad()
    
            #forward + backward + optimize
            pred = self.net(inputImage)
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
            
        #---for adam optimizer----
        #scheduler
        #self.scheduler.step()
    
            #save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': loss
            }, self.pathRes+'/checkpoint/visionTinySceneScratch_parameters_checkpoint.pt')
    
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
                inputImage, labels = batch['image'].to(self.device), batch['groundtruth'].float().to(self.device)
                pred = self.net(inputImage)
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
    
        #scheduler for SGD: for checking the improvement of val loss
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
                inputImage, labels = batch['image'].to(self.device), batch['groundtruth'].float().to(self.device)
                #forward + backward + optimize
                pred = self.net(inputImage)
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

