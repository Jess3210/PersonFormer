# -*- coding: utf-8 -*-
"""

@author: Jessica Kick
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import cv2
import numpy as np
import torchvision
import imageio #read image
from torchvggish import vggish, vggish_input, vggish_params
from transformers import BertTokenizer
from torchvggish import vggish, vggish_input
import csv
import pandas as pd
from skimage.util import random_noise
from skimage import filters
import random

def openFilePkl(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f, encoding="latin1") 
    return loaded

def saveFilePkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

class Rescale():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        video = sample
        resize = torchvision.transforms.Resize((112, 112)) #resize image to 224x224
        toPil  = transforms.ToPILImage()
        video = np.asarray(resize(toPil(video)))
        video = np.float32(video) / 255.0
        # video = np.zeros(video.shape) # 1. Experiment - Set all values to zero (black)
        # video = np.float32(video) # 1. Experiment
        # video = random_noise(video, mode='gaussian', seed=True, clip=True, var=0.1) #2. Experiment - Noise gaussian Filter
        # video = np.clip(video, 0, 1) # 2. Experiment
        # video = np.float32(video)
        # video = random_noise(video, mode='s&p', seed=True, clip=True, salt_vs_pepper=0.5, amount=0.1) #salt_vs_pepper: equal distribution salt and pepper,  #3. Experiment - salt&pepper Filter
        # video = np.clip(video, 0, 1) # 3. Experiment
        # video = np.float32(video)
        #video = filters.gaussian(video, sigma=1.5, mode='nearest', cval=0, preserve_range=False, truncate=4.0, channel_axis=-1) #4. Experiment - Blur Gaussian Filter: 1.5 sigma
        # video = filters.gaussian(video, sigma=5, mode='nearest', cval=0, preserve_range=False, truncate=4.0, channel_axis=-1) #4. Experiment - Blur Gaussian Filter: 5 sigma
        # video = np.clip(video, 0, 1) # 4. Experiment
        # video = np.float32(video)
        # video =  np.random.randint(255, size=video.shape) #random pixel
        # video = np.float32(video) / 255.0
        #video = np.float32(video)
        return video

class ToTensor():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        imagelist = []
        imageFace = sample['imageFace']
        imagePerson = sample['imagePerson']
        imageScene = sample['imageScene']
        imgFace = np.transpose(imageFace, (0, 1, 4, 2, 3)) #from B, T, W, H, C to B, T, C, H, W - notice: Batch = 1 as it is per list
        imgPerson = np.transpose(imagePerson, (0, 1, 4, 2, 3))
        imgScene = np.transpose(imageScene, (0, 1, 4, 2, 3))
        #print('Shape Imagelist: ', np.array(img).shape)
        return {'imageScene': torch.squeeze(torch.from_numpy(np.array(imgScene))), 'imageFace': torch.squeeze(torch.from_numpy(np.array(imgFace))), 'imagePerson': torch.squeeze(torch.from_numpy(np.array(imgPerson))), 'audioPerSecond': sample['audioPerSecond'], 'transcription': sample['transcription'],'groundtruth' : torch.from_numpy(np.array(sample['groundtruth'])), 'metadata': torch.from_numpy(np.array(sample['metadata'])), 'name' : sample['name']}

class Normalize():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        imageFace = sample['imageFace']
        imagePerson = sample['imagePerson']
        imageScene = sample['imageScene']
        norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #r2plus1D
        #norm = torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]) #r2plus1d as code
        #norm = torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        imageFace = norm(imageFace)
        imageFace = np.transpose(imageFace, (1, 0, 2, 3)) #from T, C, H, W to C, T, H, W
        imagePerson = norm(imagePerson)
        imagePerson = np.transpose(imagePerson, (1, 0, 2, 3)) #from T, C, H, W to C, T, H, W
        imageScene = norm(imageScene)
        imageScene = np.transpose(imageScene, (1, 0, 2, 3)) #from T, C, H, W to C, T, H, W
        return {'imageScene': imageScene, 'imageFace': imageFace,  'imagePerson' : imagePerson, 'audioPerSecond': sample['audioPerSecond'], 'transcription': sample['transcription'], 'groundtruth' : sample['groundtruth'], 'metadata': sample['metadata'], 'name' : sample['name']}

#padding for wav audio, not vggish input
class Padding():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        audioSecond = sample['audioPerSecond']
        # padd = torch.zeros(1, 1, 96, 64) # dim: 1, 1, 96, 64
        padd = torch.zeros(1, 1, 96, 64) 
        diffLen = 15 - len(audioSecond)
        
        if diffLen > 0:
            for i in range(diffLen):
                audioSecond = torch.cat((audioSecond, padd), 0)     
            
        return {'imageScene': sample['imageScene'],'imageFace': sample['imageFace'], 'imagePerson' : sample['imagePerson'],'audioPerSecond': audioSecond, 'transcription': sample['transcription'], 'groundtruth' : sample['groundtruth'], 'metadata': sample['metadata'], 'name' : sample['name']}

def getFrames(videoPath, videoName, framesPerChunk = 32, numChunks=1):
    fc = 1
    saveFrames = []
    allFrameChunks = []
    framecounter = 0
     
    for numFrame in range(1, (framesPerChunk*numChunks)+1):
        #print('Framecounter: ', numFrame)
        framecounter += 1
        readImgPath = videoPath + '/' + videoName + '_' + str(numFrame) + '.jpg'
        if(os.path.isfile(readImgPath)):
            if (fc <= framesPerChunk):
                image = imageio.imread(readImgPath)
                transform = Rescale()
                resizedFrame = transform(image)
                saveFrames.append(resizedFrame) 

                fc += 1
                
            if (fc == framesPerChunk + 1):  #after reaching the 32 frames, skip 30 frames before getting next 32 frames - stride 2
                #stack frames together for every video
                stackedFrames = np.stack(saveFrames, axis = 0)
                allFrameChunks.append(stackedFrames)
                saveFrames.clear()
                fc = 1
        else:
            break
                
    cv2.destroyAllWindows() 
    return allFrameChunks

def getAudioFromWav(audioPath, sampleRate=44100, secondsPerChunk = 3):
    vggish_params.EXAMPLE_HOP_SECONDS = secondsPerChunk
    vggish_params.SAMPLE_RATE = sampleRate
    audioChunks = vggish_params.EXAMPLE_HOP_SECONDS
    samplingRate = vggish_params.SAMPLE_RATE
    mel_spec = vggish_input.wavfile_to_examples(audioPath, secondsPerChunk)
    return mel_spec


class ChaLearnDataset(Dataset):
    def __init__(self, root_dir, video_folder, frames_folder_scene, frames_folder_face, frames_folder_person, audio_folder, transcription_file, groundtruth_file, metadata_file, transform=None):
        self.root_dir = root_dir
        self.video_folder = root_dir + video_folder
        self.frames_folder_scene = root_dir + frames_folder_scene
        self.frames_folder_face = root_dir + frames_folder_face
        self.frames_folder_person = root_dir + frames_folder_person
        self.audio_folder = root_dir + audio_folder
        self.video_files = os.listdir(self.video_folder)
        self.transcription_file = openFilePkl(root_dir + transcription_file)
        self.groundtruth_file = openFilePkl(root_dir + groundtruth_file)
        self.metadata_file = root_dir + metadata_file
        self.transform = transform

        self.metadata = []
        metadataload = []
        #metadata
        with open(self.metadata_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                metadataload.append(row)

        metapanda = pd.DataFrame(metadataload)
        metapanda = metapanda.drop([0])

        ethnicity = pd.get_dummies(metapanda[2])
        gender = pd.get_dummies(metapanda[3])

        self.metadata = (pd.concat([metapanda.drop([2, 3], axis=1), ethnicity, gender], axis=1)).values.tolist()


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        #random videos
        randomNum = random.randint(0, 1998)
        randomVid = self.video_files[randomNum]

        vid = self.video_files[idx]
        transcription = self.transcription_file[vid]
        #transcription = self.transcription_file[vid]
        extractedAudio_per_second = getAudioFromWav(self.audio_folder + '/' + vid + '.wav', secondsPerChunk=1)
        #audioPerSecond = self.netVggish(self.audio_per_second[idx])
        #personalityGT = self.personalityGT[idx]
        #name = self.name[idx]
        #lenAudio = len(audioPerSecond)

        metaEncoded = []
        # for meta in self.metadata:
        #         if meta[0] == vid:
        #             metaEncoded.extend([meta[2], meta[3], meta[4], meta[5], meta[6], 0, 0, 0]) 
        #             #metaEncoded.extend([meta[2], meta[3], meta[4], meta[5], meta[6]]) 
        
        for meta in self.metadata:
                if meta[0] == randomVid:
                    metaEncoded.extend([meta[2], meta[3], meta[4], meta[5], meta[6], 0, 0, 0]) 
                    #metaEncoded.extend([meta[2], meta[3], meta[4], meta[5], meta[6]]) 
        
        personalityTraits = [] 
        #video extraction
        extractedFramesScene = np.array(getFrames(self.frames_folder_scene, vid))
        extractedFramesFace = np.array(getFrames(self.frames_folder_face, vid))
        extractedFramesPerson = np.array(getFrames(self.frames_folder_person, vid))
        # extractedFramesScene = np.array(getFrames(self.frames_folder_scene, randomVid))
        # extractedFramesFace = np.array(getFrames(self.frames_folder_face, randomVid))
        # extractedFramesPerson = np.array(getFrames(self.frames_folder_person, randomVid))
        #print('len extracted: ', len(extractedFrames[0][0]))
        #traits groundtruth extraction per video
        personalityTraits.extend([self.groundtruth_file['extraversion'][vid], 
                                    self.groundtruth_file['agreeableness'][vid], 
                                    self.groundtruth_file['conscientiousness'][vid], 
                                    self.groundtruth_file['neuroticism'][vid], 
                                    self.groundtruth_file['openness'][vid]])
        
        sample = {'imageScene': extractedFramesScene,'imageFace': extractedFramesFace,  'imagePerson' : extractedFramesPerson, 'audioPerSecond': extractedAudio_per_second, 'transcription': transcription, 'groundtruth' : personalityTraits, 'metadata': metaEncoded, 'name' : vid}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
