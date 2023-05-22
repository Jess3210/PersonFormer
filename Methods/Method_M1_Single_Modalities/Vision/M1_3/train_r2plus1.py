# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:46:38 2022

@author: jessi
"""

import torchvision
from timeit import default_timer as timer
from fire import Fire 
#Dataloader for datatype 1 with extraction of 2 x 32 frames for 2 x 3 seconds
#from scripts_packages.dataloader_ChaLearn_datatype1 import *
#Dataloader for datatype 2 with extraction of 32 frames of complete video
from scripts_packages.dataloader_ChaLearn_datatype2 import *
from r2plus1finetuning_M1_3 import visionFinetuningClass #r2plus1 class method M1.3
#from r2plus1trainingScratch_M1_4 import visionTrainingClass as visionFinetuningClass #r2plus1 class method M1.4: Training from scratch
import pickle
        
def process(bs=16, epochs=100, lossfunction='mse', learningrate=1e-5, cudadevice='cuda', root_dir_path = './ChaLearn_First_Impression/'):
    composed = torchvision.transforms.Compose([ToTensor(), Normalize()])

    trainset = ChaLearnDataset(root_dir=root_dir_path+'train/', video_folder='train_video', frames_folder='train_frames', groundtruth_file='train_groundtruth/annotation_training.pkl', transform=composed)
    valset = ChaLearnDataset(root_dir=root_dir_path+'val/', video_folder='val_video', frames_folder='val_frames', groundtruth_file='val_groundtruth/annotation_validation.pkl', transform=composed)
    #testset = ChaLearnDataset(root_dir=root_dir_path+'test/', video_folder='test_video', frames_folder='test_frames', groundtruth_file='test_groundtruth/annotation_test.pkl', transform=composed)

    train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True) #, num_workers=self.num_workers)
    val_dl = torch.utils.data.DataLoader(valset, batch_size=bs,
                                             shuffle=False) #, num_workers=self.num_workers)
    # test_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
    #                                          shuffle=False) #, num_workers=self.num_workers)
    
    modelProcess = visionFinetuningClass(optimizer='sgd', lossfnct=lossfunction, learningrate=learningrate, device=cudadevice)
    
    #for loading checkpoint during a training braak
    # checkpointLoader = torch.load('visionFinetuning/checkpoint/visionFinetuning_parameters_checkpoint.pt')
    # epochsCheck = checkpointLoader['epoch']

    timer_counter = 0
    
    for epoch in range(epochs):
        #train
        print('Epoch: ', epoch+1)
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