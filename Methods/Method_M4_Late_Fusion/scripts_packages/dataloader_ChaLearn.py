# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:52:00 2022

@author: jessi
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
        return video

class ToTensor():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        imagelist = []
        image = sample['image']
        img = np.transpose(image, (0, 3, 1, 2)) #from T, W, H, C to T, C, H, W
        #print('Shape Imagelist: ', np.array(img).shape)
        return {'image': torch.from_numpy(np.array(img)), 'transcription': sample['transcription'], 'audio': sample['audio'], 'groundtruth' : torch.from_numpy(np.array(sample['groundtruth'])), 'name' : sample['name']}

class Normalize():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        image = sample['image']
        norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = norm(image)
        image = np.transpose(image, (1, 0, 2, 3)) #from T, C, H, W to C, T, H, W
        return {'image': image, 'transcription': sample['transcription'], 'audio': sample['audio'], 'groundtruth' : sample['groundtruth'], 'name' : sample['name']}

def getFrames(videoPath, videoName, framesPerChunk = 32, numChunks=2):
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
                #print('Saved Frames: ', len(saveFrames))
                #stack frames together for every video
                stackedFrames = np.stack(saveFrames, axis = 0)
                #print('Stacked Frames: ', stackedFrames.shape)
                allFrameChunks.append(stackedFrames)
                #print('Shape all frame chunks: ', np.array(allFrameChunks).shape)
                saveFrames.clear()
                fc = 1
        else:
            break
                
    #print('Num of frames: ', fc)
    #print('Shape of All Frame Chunks: ', np.array(allFrameChunks).shape)
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
    def __init__(self, root_dir, video_folder, frames_folder, audio_folder, transcription_file, groundtruth_file, transform=None):
        self.root_dir = root_dir
        self.video_folder = root_dir + video_folder
        self.frames_folder = root_dir + frames_folder
        self.audio_folder = root_dir + audio_folder
        self.video_files = os.listdir(self.video_folder)
        self.transcription_file = openFilePkl(root_dir + transcription_file)
        self.groundtruth_file = openFilePkl(root_dir + groundtruth_file)
        self.transform = transform
        
        counter = 0
        self.video = []
        self.transcription = []
        self.groundtruth = []
        self.audio = []
        #self.audio_per_second = []
        self.personalityGT = []
        self.name = []
        
        # self.netVggish = vggish()
        # self.netVggish.to('cpu')
        # self.netVggish.eval()

        #self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        for vid in self.video_files:
            personalityTraits = []
            chunksCounter = 0
            
            #video extraction
            extractedFrames = np.array(getFrames(self.frames_folder, vid))
            #audio extraction as mel spec
            extractedAudio = getAudioFromWav(self.audio_folder + '/' + vid + '.wav')
            #extractedAudioWav = getAudioFromWav(self.audio_folder + '/' + vid + '.wav')
            #extractedAudio = self.netVggish(extractedAudioWav)
            #extractedAudio_per_second = getAudioFromWav(self.audio_folder + '/' + vid + '.wav', secondsPerChunk=1)
            #extractedAudio_per_second = self.netVggish(extractedAudio_per_second_wav)

            #Transcription
            #self.transcription.append(self.transcription_file[vid])

            #self.transcription.append(self.tokenizer(self.transcription_file[vid], return_tensors="pt"))

            #traits groundtruth extraction per video
            personalityTraits.extend([self.groundtruth_file['extraversion'][vid], 
                                      self.groundtruth_file['agreeableness'][vid], 
                                      self.groundtruth_file['conscientiousness'][vid], 
                                      self.groundtruth_file['neuroticism'][vid], 
                                      self.groundtruth_file['openness'][vid]])
            
            for chunks in extractedFrames: #save information for every frame chunk
                #video
                self.video.append(chunks)
                #Transcription
                self.transcription.append(self.transcription_file[vid])
                self.personalityGT.append(personalityTraits)
                self.name.append(vid + '_' + str(chunksCounter))

                self.audio.append(extractedAudio[chunksCounter])
                #self.audio_per_second.append(extractedAudio_per_second)

                chunksCounter += 1
            
            counter += 1

        #print(len(self.video_files), len(self.transcription_file.keys()), len(self.groundtruth_file['extraversion']))
        
        #assert(len(self.video_files) == len(self.transcription_file.keys()))   
        #assert(len(self.transcription_file == len(self.groundtruth_file)))

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
            #idx = idx.tolist()
        
        video = self.video[idx]
        transcription = self.transcription[idx]
        audio = self.audio[idx]
        #audioPerSecond = self.netVggish(self.audio_per_second[idx])
        personalityGT = self.personalityGT[idx]
        name = self.name[idx]
        #lenAudio = len(audioPerSecond)
        
        sample = {'image': video, 'transcription': transcription, 'audio': audio, 'groundtruth' : personalityGT, 'name' : name}
        
        if self.transform:
            sample = self.transform(sample)
        return sample

#if __name__ == '__main__':
    #composed = torchvision.transforms.Compose([ToTensor(), Normalize()])
    
    #train = ChaLearnDataset(r'C:/Users/jessi/Documents/Studium/Master_Thesis/Chalearn_First_impressoin_v2/train/', 'example_video', 'ffmpeg_example_frames', 'train_transcription/transcription_training.pkl', 'train_groundtruth/annotation_training.pkl', composed)
    #val = ChaLearnDataset(r'C:/Users/jessi/Documents/Studium/Master_Thesis/Chalearn_First_impressoin_v2/val/', 'val_video', 'val_transcription/transcription_validation.pkl', 'val_groundtruth/annotation_validation.pkl', composed)
    
    #print(train[5])
    
    #save data as pickle
    #saveFilePkl('TraindataWithFrames.pkl', train)
    #saveFilePkl('ValdataWithFrames.pkl', train)