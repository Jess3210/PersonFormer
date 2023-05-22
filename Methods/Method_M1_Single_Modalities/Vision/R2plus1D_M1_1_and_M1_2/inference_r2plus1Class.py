# -*- coding: utf-8 -*-
"""

@author: Jessica Kick
"""

import torchvision
from timeit import default_timer as timer
from fire import Fire 
from scripts_packages.dataloader_ChaLearn import *
from r2plusClass import r2plus1 #r2plus1 class
import pickle
        
def process(bs=4, epochs=1, lossfunction='mse', modelPretrained='34insta', deviceCuda='cuda', root_dir=r'/ChaLearn_First_Impression/test/', video_folder='test_video', frames_folder='test_frames', groundtruth_file='test_groundtruth/annotation_test.pkl'):
    composed = torchvision.transforms.Compose([ToTensor(), Normalize()])
    valset = ChaLearnDataset(root_dir, video_folder, frames_folder, groundtruth_file, composed)

    val_dl = torch.utils.data.DataLoader(valset, batch_size=bs,
                                             shuffle=False) #, num_workers=self.num_workers)
    
    modelProcess = r2plus1(lossfnct=lossfunction, model=modelPretrained, device=deviceCuda)
    
    time_start = timer() #measure time
    
    modelProcess.test(val_dl, epochs)

    time_end = timer() #measure time
    delta_time = time_end - time_start #delta time   
    print('Finished Training')
    print('Prediction Time: ', delta_time)

if __name__ == '__main__':
    Fire(process)