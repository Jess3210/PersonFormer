# -*- coding: utf-8 -*-
"""

@author: Jessica Kick
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import numpy as np

def openFilePkl(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f, encoding="latin1") 
    return loaded

def saveFilePkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

class ChaLearnDataset(Dataset):
    def __init__(self, root_dir, video_folder, transcription_file, groundtruth_file):
        self.root_dir = root_dir
        self.video_folder = root_dir + video_folder
        self.video_files = os.listdir(self.video_folder)
        self.transcription_file = openFilePkl(root_dir + transcription_file)
        self.groundtruth_file = openFilePkl(root_dir + groundtruth_file)
        
        counter = 0
        self.video = []
        self.transcription = []
        self.groundtruth = []
        self.personalityGT = []
        self.name = []
        
        for vid in self.video_files:
            personalityTraits = []
            chunksCounter = 0
            
            self.transcription.append(self.transcription_file[vid])
        
            #traits groundtruth extraction per video
            personalityTraits.extend([self.groundtruth_file['extraversion'][vid], 
                                      self.groundtruth_file['agreeableness'][vid], 
                                      self.groundtruth_file['conscientiousness'][vid], 
                                      self.groundtruth_file['neuroticism'][vid], 
                                      self.groundtruth_file['openness'][vid]])
            
            self.personalityGT.append(personalityTraits)
            
            counter += 1

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        transcription = self.transcription[idx]
        personalityGT = self.personalityGT[idx]
        
        sample = {'transcription': transcription, 'groundtruth' : torch.from_numpy(np.array(personalityGT))}
        
        return sample