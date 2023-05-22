# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:46:38 2022

@author: jessi
"""

from torchvision import transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from timeit import default_timer as timer
from fire import Fire 
from scripts_packages.dataloader_ChaLearn import *  #dataloader
from visionAudioClass import visionAudioClass 
import pickle
        
def process(bs=32, epochs=50, lossfunction='mse', learningrate=1e-5, cudadevice='cuda', root_dir_path = './ChaLearn_First_Impression/'):
    composed = torchvision.transforms.Compose([ToTensor(), Normalize()])
    
    trainset = ChaLearnDataset(root_dir_path+'train/', 'train_video', 'train_frames', 'train_audio', 'train_transcription/transcription_training.pkl', 'train_groundtruth/annotation_training.pkl', composed)
    valset = ChaLearnDataset(root_dir_path+'val/', 'val_video', 'val_frames', 'val_audio', 'val_transcription/transcription_validation.pkl', 'val_groundtruth/annotation_validation.pkl', composed)
    #testset = ChaLearnDataset(root_dir_path+'test/', 'test_video', 'test_frames', 'test_audio', 'test_transcription/transcription_test.pkl', 'test_groundtruth/annotation_test.pkl', composed)


    train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True) #, num_workers=self.num_workers)
    val_dl = torch.utils.data.DataLoader(valset, batch_size=bs,
                                             shuffle=False) #, num_workers=self.num_workers)
    # test_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
    #                                          shuffle=False) #, num_workers=self.num_workers)
    
    modelProcess = visionAudioClass(optimizer='sgd', lossfnct=lossfunction, learningrate=learningrate, device=cudadevice)
    
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
