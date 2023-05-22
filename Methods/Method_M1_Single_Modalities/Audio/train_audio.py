# -*- coding: utf-8 -*-
"""
@author: Jessica Kick
"""

import torchvision
from timeit import default_timer as timer
from fire import Fire 
#Dataloader for datatype 1 with extraction of 2 x 32 frames for 2 x 3 seconds
#from scripts_packages.dataloader_ChaLearn_datatype1 import *
#Dataloader for datatype 2 with extraction of 32 frames of complete video
from scripts_packages.dataloader_ChaLearn_datatype2 import *
from audioClass_with_transformer_flattenWithPosEnc_M1_14 import audioClass 
#from audioClass_with_transformer_mean_posEnc_M1_13 import audioClass 
#from audioClass_trained_mean_M1_8_till_M1_12 import audioClass 

import pickle
        
#def process(bs=32, epochs=50, learningrate=1e-5, cudadevice='cuda:0'):
def process(bs=32, epochs=30, learningrate=1e-5, cudadevice='cuda:0', root_dir_path = './ChaLearn_First_Impression/'):
    composed = torchvision.transforms.Compose([ToTensor(), Padding()]) #just for datatype2
    #composed = torchvision.transforms.Compose([ToTensor()]) #just for datatype1
    
    trainset = ChaLearnDataset(root_dir_path+'train/', 'train_video', 'train_frames', 'train_audio', 'train_transcription/transcription_training.pkl', 'train_groundtruth/annotation_training.pkl', composed)
    valset = ChaLearnDataset(root_dir_path+'val/', 'val_video', 'val_frames', 'val_audio', 'val_transcription/transcription_validation.pkl', 'val_groundtruth/annotation_validation.pkl', composed)
    #testset = ChaLearnDataset(rroot_dir_path+'test/', 'test_video', 'test_frames', 'test_audio', 'test_transcription/transcription_test.pkl', 'test_groundtruth/annotation_test.pkl', composed)

    train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True) #, num_workers=self.num_workers)
    val_dl = torch.utils.data.DataLoader(valset, batch_size=bs,
                                             shuffle=False) #, num_workers=self.num_workers)
    # test_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
    #                                          shuffle=False) #, num_workers=self.num_workers)
    
    modelProcess = audioClass(optimizer='sgd', lossfnct=lossfunction, learningrate=learningrate, device=cudadevice bs = bs)
   
    # checkpointLoader = torch.load('audioMean/checkpoint/audio_parameters_checkpoint.pt')
    # epochsCheck = checkpointLoader['epoch']

    timer_counter = 0
    
    for epoch in range(epochs):
        #train
        time_start = timer() #measure time
        modelProcess.train(train_dl, epoch)
        #val
        modelProcess.val(val_dl, epoch)
        time_end = timer() #measure time
        delta_time = time_end - time_start #delta time
        timer_counter += delta_time
 
        #test
        #modelProcess.test(test_dl, epoch)


    print('Finished Training')
    print('Prediction Time Train for all epochs: ', timer_counter)

    modelProcess.saveModelResults(delta_time = timer_counter)

if __name__ == '__main__':
    Fire(process)