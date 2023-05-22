# -*- coding: utf-8 -*-
"""

@author: Jessica Kick
"""

import torch
from timeit import default_timer as timer
from fire import Fire 
from scripts_packages.dataloader_text_finetuning import *  #dataloader
from textClassTrain import textClass 
        
def process(bs=8, epochs=2, learningrate=1e-5, cudadevice='cuda:0', root_dir_path = './ChaLearn_First_Impression/'):
    
    trainset = ChaLearnDataset(root_dir_path+'train/', 'train_video', 'train_transcription/transcription_training.pkl', 'train_groundtruth/annotation_training.pkl')
    valset = ChaLearnDataset(root_dir_path+'val/', 'val_video', 'val_transcription/transcription_validation.pkl', 'val_groundtruth/annotation_validation.pkl')
    #testset = ChaLearnDataset(root_dir_path+'test/', 'test_video', 'test_transcription/transcription_test.pkl', 'test_groundtruth/annotation_test.pkl')


    train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=False) #, num_workers=self.num_workers)
    val_dl = torch.utils.data.DataLoader(valset, batch_size=bs,
                                             shuffle=False) #, num_workers=self.num_workers)
    # test_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
    #                                          shuffle=False) #, num_workers=self.num_workers)
    
    modelProcess = textClass(optimizer='sgd', lossfnct='mse', learningrate=1e-5, device=cudadevice)
    
    #---load checkpoint after a break of training ---
    #checkpointLoader = torch.load('visionAudioText/checkpoint/visionAudioText_parameters_checkpoint.pt')
    #epochsCheck = checkpointLoader['epoch']

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
    
    #modelProcess.train(train_dl, epochs)
    #modelProcess.val(val_dl, epochs)

    print('Finished Training')
    print('Prediction Time Train for all epochs: ', timer_counter)

    modelProcess.saveModelResults(delta_time = timer_counter)

if __name__ == '__main__':
    Fire(process)
