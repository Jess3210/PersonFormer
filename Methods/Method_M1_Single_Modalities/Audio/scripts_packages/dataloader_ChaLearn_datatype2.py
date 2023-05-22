# -*- coding: utf-8 -*-
"""

@author: Jessica Kick
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import cv2
import numpy as np
import torchvision
import imageio #read image
from torchvggish import vggish, vggish_input, vggish_params
from torchvggish import vggish, vggish_input

def openFilePkl(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f, encoding="latin1") 
    return loaded

def saveFilePkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

class ToTensor():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        return {'audioPerSecond': sample['audioPerSecond'], 'groundtruth' : torch.from_numpy(np.array(sample['groundtruth'])), 'LenAudioSecond' : sample['LenAudioSecond']}

#padding for wav audio, not vggish input
class Padding():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        audioSecond = sample['audioPerSecond']
        padd = torch.zeros(1, 1, 96, 64) 
        diffLen = 15 - len(audioSecond)
        
        if diffLen > 0:
            for i in range(diffLen):
                audioSecond = torch.cat((audioSecond, padd), 0)     
            
        return {'audioPerSecond': audioSecond, 'groundtruth' : sample['groundtruth'],  'LenAudioSecond' : sample['LenAudioSecond']}

def getAudioFromWav(audioPath, sampleRate=44100, secondsPerChunk = 3):
    vggish_params.EXAMPLE_HOP_SECONDS = secondsPerChunk
    vggish_params.SAMPLE_RATE = sampleRate
    audioChunks = vggish_params.EXAMPLE_HOP_SECONDS
    samplingRate = vggish_params.SAMPLE_RATE
    mel_spec = vggish_input.wavfile_to_examples(audioPath, secondsPerChunk)
    return mel_spec

class ChaLearnDataset(Dataset):
    def __init__(self, root_dir, video_folder, frames_folder, audio_folder, transcription_file, groundtruth_file, transform=None):
        self.root_dir = root_dir
        self.video_folder = root_dir + video_folder
        self.audio_folder = root_dir + audio_folder
        self.video_files = os.listdir(self.video_folder)
        self.groundtruth_file = openFilePkl(root_dir + groundtruth_file)
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        vid = self.video_files[idx]
        personalityTraits = []
        
        extractedAudio_per_second = getAudioFromWav(self.audio_folder + '/' + vid + '.wav', secondsPerChunk=1) #select spectrogram extractions from seconds 
        personalityTraits.extend([self.groundtruth_file['extraversion'][vid], 
                                  self.groundtruth_file['agreeableness'][vid], 
                                  self.groundtruth_file['conscientiousness'][vid], 
                                  self.groundtruth_file['neuroticism'][vid], 
                                  self.groundtruth_file['openness'][vid]])
        
        lenAudio = len(extractedAudio_per_second)
        
        sample = {'audioPerSecond': extractedAudio_per_second, 'groundtruth' : personalityTraits,  'LenAudioSecond' : lenAudio}
        
        if self.transform:
            sample = self.transform(sample)
        return sample