import torch
import torch.nn as nn
import torchvision.models as models

from scripts_packages.metricLoss import * #mean average accuracy, average accuracy, mse per trait

class r2plus1():
    def __init__(self, lossfnct='mse', model='34insta', device='cuda'):
        #Model R2plus1D - 18 pretrained with kineteics
        if(model == '18kin'):
            self.net = models.video.r2plus1d_18(pretrained=True)
        #Model R2plus1D - 34 pretrained with kineticss
        elif(model == '34kin'):
            print('pretrained 34 kinetic')
            self.net = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=True)
        else:
        #Model R2plus1D - 34 pretrained with Instagram data
            print('pretrained 34 insta dataset')
            self.net = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True)
        
        self.criterion = nn.MSELoss()
        self.classes = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        # Freeze early layers
        for param in self.net.parameters():
            param.requires_grad = False
        
        n_inputs = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(512, len(self.classes), bias=True), nn.Sigmoid())

        self.net.to(device)
        
    
    def test(self, test_dl, epoch):
        self.net.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        mseClassEpoch = torch.zeros(1, len(self.classes)).to(self.device)
        traitAccPerBatch = torch.zeros(1, len(self.classes)).to(self.device)

        with torch.no_grad():
            for batch in test_dl:
                inputs, labels = batch['image'].to(self.device), batch['groundtruth'].float().to(self.device)

                pred = self.net(inputs)
                #loss mse
                loss = self.criterion(pred, labels) #MSE Loss
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
        
        print('Average Accuracy per Trait: ', traitAccPerBatch / len(test_dl))
        meanAvgAccPerEpoch = meanAverageAccuracy(traitAccPerBatch / len(test_dl), len(self.classes))
        print('Mean Average Accuracy: ', meanAvgAccPerEpoch)
        print('Loss Training: {}'.format(running_loss / len(test_dl)))       
        print('Loss per Class Training: {}'.format(torch.div(mseClassEpoch, len(test_dl))))    
