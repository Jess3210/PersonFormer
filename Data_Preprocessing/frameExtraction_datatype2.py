# -*- coding: utf-8 -*-
"""

@author: Jessica Kick
"""

import ffmpeg
import cv2
import random
from timeit import default_timer as timer
import os
import numpy as np
import pickle
from fire import Fire

def ffmpegFrameExtraction(videoPath, imagePath, stride=4, chunks = 1, seconds = 15, framesPerChunk=32):
    fileNames = os.listdir(videoPath)
    time_start = timer() #measure time
    
    for videos in fileNames:
        probe = ffmpeg.probe(videoPath + '/' + videos)
        #width = 224
        width = probe['streams'][0]['width']
        #outputPathImage = 'ffmpeg_example_frames/'
        fps = int(probe['streams'][0]['nb_frames']) / float(probe['streams'][0]['duration'])

        time = int(float(probe['streams'][0]['duration']))

        if time < (seconds):
            seconds = time
        
        counterFrames = 1

        framesPerSecond = int(framesPerChunk / seconds)
        lastSecond = framesPerChunk - ((seconds - 1) * framesPerSecond)

        frameNum = []

        for i in range(chunks):
            currentFrames = 1
            for j in range(seconds*i, (seconds*i)+seconds):
                if j == (seconds*i+seconds - 1):
                    frameNum.extend(np.arange(j*fps, (j*fps)+(lastSecond*stride), stride)) 

                else:
                    frameNum.extend(np.arange(j*fps, (j*fps)+(framesPerSecond*stride), stride)) 

        #image frames extraction
        for selectedFrame in frameNum:
            ffmpeg.input(videoPath + '/' + videos).filter('select', 'gte(n, {})'.format(selectedFrame)).filter('scale', width, -1).output(imagePath + '/' + videos + '_' + str(counterFrames) + '.jpg', vframes=1).run()
            counterFrames += 1
   
    time_end = timer() #measure time
    delta_time = time_end - time_start #delta time

if __name__ == '__main__':
    ffmpegFrameExtraction(r'./ChaLearn_First_Impression/val/val_video', r'./ChaLearn_First_Impression/val/val_frames_all')

