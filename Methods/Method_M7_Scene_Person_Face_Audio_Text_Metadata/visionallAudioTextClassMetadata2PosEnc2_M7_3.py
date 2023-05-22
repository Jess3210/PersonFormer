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

from torchvggish import vggish, vggish_input
from architectures.vggish import VGGish
from transformers import BertConfig, BertModel, BertTokenizer

from architectures.r2plus1finetuning import visionClass
from architectures.textFinetuning import textFinetuning as textClass
from architectures.audioClass_1s import audioConcat as audioClass

class visionAudioTextFusion(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        faceCheckpoint = './checkpoint/r2plus1_scene_parameters_checkpoint.pth', #load checkpoints - needs to be added in folder
        personCheckpoint = './checkpoint/r2plus1_scene_parameters_checkpoint.pth', #load checkpoints - needs to be added in folder
        sceneCheckpoint = './checkpoint/r2plus1_scene_parameters_checkpoint.pth', #load checkpoints - needs to be added in folder
        textCheckpoint = './checkpoint/text_parameters_finetuning.pth',
        audio_model = VGGish(torch.load('./checkpoint/pytorch_vggish.pth')), #no pretrained weights: VGGish(None)
        input_vision=768,
        output_vision=512,
        d_model2=128,
        d_model=512,
        nhead = 8,
        dimFeedforward = 2048,
        dimFeedforward1 = 1024, #audio feed forward
        numLayers = 2,
        dropoutTransformer=0.1,
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

        self.audio_module = audio_model.to(torch.device('cuda:0'))

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

        self.vision_module_face.vision_module.fc = nn.Linear(output_vision, output_vision, bias=True)
        self.vision_module_person.vision_module.fc = nn.Linear(output_vision, output_vision, bias=True)
        self.vision_module_scene.vision_module.fc = nn.Linear(output_vision, output_vision, bias=True)

        #self.audio_module.fc = nn.Linear(128, 128, bias=True)
        self.text_module.fc = nn.Linear(768, 768, bias=True)

        self.pos_encoder_audio = PositionalEncoding(d_model=d_model2, dropout=dropoutTransformer)

        #layer1
        self.layer_norm1 = nn.LayerNorm(d_model2)
        self.self_attn1 = nn.MultiheadAttention(d_model2, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer1_1 = nn.Linear(d_model2, dimFeedforward1)
        self.fctransformer1_2 = nn.Linear(dimFeedforward1, d_model2)

        #layer2
        self.layer_norm1 = nn.LayerNorm(d_model2)
        self.self_attn2 = nn.MultiheadAttention(d_model2, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer2_1 = nn.Linear(d_model2, dimFeedforward1)
        self.fctransformer2_2 = nn.Linear(dimFeedforward1, d_model2)

        self.pooling = nn.AdaptiveAvgPool2d((1,128))

        #get fusion layer from 1029 to 512
        self.FacePersonlayer = torch.nn.Linear( 
            in_features=(output_vision+output_vision+5), #face 512 + person 512 + metadata 5
            out_features=(output_vision)
        )

        #person + face transformer
        #layer1
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer3_1 = nn.Linear(d_model, dimFeedforward)
        self.fctransformer3_2 = nn.Linear(dimFeedforward, d_model)

        #layer2
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer4_1 = nn.Linear(d_model, dimFeedforward)
        self.fctransformer4_2 = nn.Linear(dimFeedforward, d_model)

        self.fusion_avt = torch.nn.Linear(
            #in_features=(language_feature_dim + vision_feature_dim), 
            in_features=(768+128+512+5), #fusion audio + text + vision + metadata 5
            out_features=512
        )

        #scene + text + audio
        #layer1
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn5 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer5_1 = nn.Linear(d_model, dimFeedforward)
        self.fctransformer5_2 = nn.Linear(dimFeedforward, d_model)

        #layer2
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn6 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer6_1 = nn.Linear(d_model, dimFeedforward)
        self.fctransformer6_2 = nn.Linear(dimFeedforward, d_model)

        self.fusion_avtfp_hidden = torch.nn.Linear(
            #in_features=(language_feature_dim + vision_feature_dim), 
            #in_features=(1024+512), 
            in_features=1024, #1024
            out_features=512
        )
        #scene + text + audio
        #layer1
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn7 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer7_1 = nn.Linear(d_model, dimFeedforward)
        self.fctransformer7_2 = nn.Linear(dimFeedforward, d_model)

        #layer2
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn8 = nn.MultiheadAttention(d_model, nhead, dropout=dropoutTransformer, batch_first=True)
        self.fctransformer8_1 = nn.Linear(d_model, dimFeedforward)
        self.fctransformer8_2 = nn.Linear(dimFeedforward, d_model)
        
        self.output = torch.nn.Linear(
            in_features=512, 
            out_features=num_classes
        )
        self.dropoutTransf = torch.nn.Dropout(dropoutTransformer)
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, imageFace, imagePerson, imageScene, audio, text, metadata): 
        visionFace = self.vision_module_face(imageFace)
        visionPerson = self.vision_module_person(imagePerson)
        visionScene = self.vision_module_scene(imageScene)
        textfeat = self.text_module(text)

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

        audio_features = torch.squeeze(self.pooling(audio_features))

        #projection layer person + face
        combinedPersonFace = torch.cat(
            [visionFace, visionPerson, metadata], dim=1
        )

        #projection layer scene + audio + text
        combinedVAT = torch.cat(
            [visionScene, audio_features, textfeat, metadata], dim=1
        )
        #hidden layer scene + audio + text
        fusedAVT = self.dropout(
            torch.nn.functional.relu(
                self.fusion_avt(combinedVAT)
            )
        )

        residual = fusedAVT
        fusedAVT_features = self.self_attn5(fusedAVT, fusedAVT, fusedAVT)[0]
        fusedAVT_features += residual
        fusedAVT_features = self.layer_norm(fusedAVT_features)
        residual = fusedAVT_features
        fusedAVT_features = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer5_1(fusedAVT_features)))
        fusedAVT_features = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer5_2(fusedAVT_features)))
        fusedAVT_features += residual
        fusedAVT_features = self.layer_norm(fusedAVT_features)

        #layer2 Transformer
        residual = fusedAVT_features
        fusedAVT_features = self.self_attn6(fusedAVT_features, fusedAVT_features, fusedAVT_features)[0]
        fusedAVT_features += residual
        fusedAVT_features = self.layer_norm(fusedAVT_features)
        residual = fusedAVT_features
        fusedAVT_features = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer6_1(fusedAVT_features)))
        fusedAVT_features = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer6_2(fusedAVT_features)))
        fusedAVT_features += residual
        fusedAVT_features = self.layer_norm(fusedAVT_features)


        #hidden layer face + person
        hiddenLayerFacePerson = self.dropout(
            torch.nn.functional.relu(
                self.FacePersonlayer(combinedPersonFace)
            )
        )

        residual = hiddenLayerFacePerson
        facePerson_features = self.self_attn3(hiddenLayerFacePerson, hiddenLayerFacePerson, hiddenLayerFacePerson)[0]
        facePerson_features += residual
        facePerson_features = self.layer_norm(facePerson_features)
        residual = facePerson_features
        facePerson_features = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer3_1(facePerson_features)))
        facePerson_features = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer3_2(facePerson_features)))
        facePerson_features += residual
        facePerson_features = self.layer_norm(facePerson_features)

        #layer2 Transformer
        residual = facePerson_features
        facePerson_features = self.self_attn4(facePerson_features, facePerson_features, facePerson_features)[0]
        facePerson_features += residual
        facePerson_features = self.layer_norm(facePerson_features)
        residual = facePerson_features
        facePerson_features = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer4_1(facePerson_features)))
        facePerson_features = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer4_2(facePerson_features)))
        facePerson_features += residual
        facePerson_features = self.layer_norm(facePerson_features)

        #projection layer scene + audio + text + metadata
        combinedVATFP = torch.cat(
            [fusedAVT_features, facePerson_features], dim=1
        )
        hiddenLayerVATFP = self.dropout(
            torch.nn.functional.relu(
                self.fusion_avtfp_hidden(combinedVATFP)
            )
        )

        residual = hiddenLayerVATFP
        features_all = self.self_attn7(hiddenLayerVATFP, hiddenLayerVATFP, hiddenLayerVATFP)[0]
        features_all += residual
        features_all = self.layer_norm(features_all)
        residual = features_all
        features_all = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer7_1(features_all)))
        features_all = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer7_2(features_all)))
        features_all += residual
        features_all = self.layer_norm(features_all)

        #layer2 Transformer
        residual = features_all
        features_all = self.self_attn8(features_all, features_all, features_all)[0]
        features_all += residual
        features_all = self.layer_norm(features_all)
        residual = features_all
        features_all = self.dropoutTransf(torch.nn.functional.relu(
            self.fctransformer8_1(features_all)))
        features_all = self.dropoutTransf(
            torch.nn.functional.relu(self.fctransformer8_2(features_all)))
        features_all += residual
        features_all = self.layer_norm(features_all)

        logits = self.output(features_all)
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
        embedding = torch.permute(embedding, (1, 0, 2))
        embedding = embedding + self.pe[:embedding.size(0)]
        drop = self.dropout(embedding)
        outputEmbedding = torch.permute(drop, (1, 0, 2))
        return(outputEmbedding)

class visionAudioTextFusionClass():
    def __init__(self, optimizer='sgd', lossfnct='mse', model='inst', learningrate=1e-5, device='cuda'):

        self.pathRes = './visionAudioTextM2TransformerVisionAudio'
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

        checkpointLoader = torch.load('visionAudioTextM2TransformerVisionAudio/checkpoint/visionAudioTextPersonFaceFusion_parameters_checkpoint.pt')
        self.net.load_state_dict(checkpointLoader['model_state_dict'])
        self.optim.load_state_dict(checkpointLoader['optimizer_state_dict'])

        #send optimizer to GPU
        #self.optimizer_to(self.optim, self.device)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=3, factor=0.1)
        
        self.net.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

        # self.loss_values_train = openFilePkl('visionAudioTextM2TransformerVisionAudio/lossValuesTrain.pickle')
        # self.loss_values_val = openFilePkl('visionAudioTextM2TransformerVisionAudio/lossValuesVal.pickle')
        # #self.loss_values_test = openFilePkl('visionAudioTextM1/lossValuesTest.pickle')
        # self.mse_class_train = openFilePkl('visionAudioTextM2TransformerVisionAudio/mseClassTrain.pickle')
        # self.mse_class_val = openFilePkl('visionAudioTextM2TransformerVisionAudio/mseClassVal.pickle')
        # #self.mse_class_test = openFilePkl('visionAudioTextM1/mseClassTest.pickle')
        # self.traitAccTrain = openFilePkl('visionAudioTextM2TransformerVisionAudio/traitAccTrain.pickle')
        # self.meanAvgAccAllTrain = openFilePkl('visionAudioTextM2TransformerVisionAudio/meanAvgAccAllTrain.pickle')
        # self.traitAccVal = openFilePkl('visionAudioTextM2TransformerVisionAudio/traitAccVal.pickle')
        # self.meanAvgAccAllVal = openFilePkl('visionAudioTextM2TransformerVisionAudio/meanAvgAccAllVal.pickle')
                
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
            inputImageFace, inputImagePerson, inputImageScene, audio, text, labels, metadata, name = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['imageScene'].to(self.device), batch['audioPerSecond'].to(self.device), inputTextToken.to(self.device), batch['groundtruth'].float().to(self.device), batch['metadata'].to(self.device), batch['name']
            self.optim.zero_grad()
    
            #forward + backward + optimize
            pred = self.net(inputImageFace, inputImagePerson, inputImageScene, audio, text, metadata)
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
                inputImageFace, inputImagePerson, inputImageScene, audio, text, labels, metadata, name = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['imageScene'].to(self.device), batch['audioPerSecond'].to(self.device), inputTextToken.to(self.device), batch['groundtruth'].float().to(self.device), batch['metadata'].to(self.device), batch['name']
                pred = self.net(inputImageFace, inputImagePerson, inputImageScene, audio, text, metadata)
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
                inputImageFace, inputImagePerson, inputImageScene, audio, text, labels, metadata, name = batch['imageFace'].to(self.device), batch['imagePerson'].to(self.device), batch['imageScene'].to(self.device), batch['audioPerSecond'].to(self.device), inputTextToken.to(self.device), batch['groundtruth'].float().to(self.device), batch['metadata'].to(self.device), batch['name']
                #forward + backward + optimize
                pred = self.net(inputImageFace, inputImagePerson, inputImageScene, audio, text, metadata)
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
