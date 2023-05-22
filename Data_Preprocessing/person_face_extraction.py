# -*- coding: utf-8 -*-
"""

@author: Jessica Kick
copyright
"""

from ultralytics import YOLO
import os
import cv2
from timeit import default_timer as timer
import pickle
import numpy as np
from retinaface import RetinaFace

def openFilePkl(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f, encoding="latin1") 
    return loaded

def saveFilePkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def personPrediction(pathFile, saveFileName):
    fileNames = os.listdir(pathFile)
    
    modelPerson = YOLO("yolov8s.pt") #download yolov8s.pt weigth from: https://github.com/ultralytics/ultralytics
    resPerson = []
    
    #---Person Detection ------
    time_start = timer() #measure time
    for img in fileNames:
        pred = modelPerson.predict(pathFile+img) #, conf=0.5)
        predictions = []
        for i in pred:   
            if 'det' in i:
                for j in list(i['det']):
                    if j[5] == 0:
                        predictions.append(j)   
            else:
                predictions.append('none')
    
        resPerson.append({'result': predictions, 'name': img})
    time_end = timer() #measure time
    delta_time = time_end - time_start #delta time
    
    print('Time for Prediction Person: ', delta_time)
    
    saveFilePkl(saveFileName + 'personBbx.pickle', resPerson)
    saveFilePkl(saveFileName + 'personTime.pickle', delta_time)
    
    return resPerson

def facePrediction(pathFile, saveFileName):
    fileNames = os.listdir(pathFile)
    resFace = []
    counter = 0
    
    #---Face detection ---
    time_start = timer() #measure time
    for img in fileNames:
        pred = RetinaFace.detect_faces(pathFile+img) #, conf=0.5)
        predictions = []
        for i in pred:
            if len(i) != 0:
                predictions.append(pred[i]['facial_area'])
            else:
                predictions.append(0)
    
        resFace.append({'result': predictions, 'name': img})
        
    time_end = timer() #measure time
    delta_time = time_end - time_start #delta time
    
    print('Time for Prediction Face: ', delta_time)
    
    saveFilePkl(saveFileName + 'faceBbx.pickle', resFace)
    saveFilePkl(saveFileName + 'faceTime.pickle', delta_time)
    
    return resFace

def drawBbxWithCutPerson(res, pathReadFrames, pathSaveFrames):
    for detect in res:
        if detect['result'] == 'none' or len(detect['result']) == 0 or detect['result'][0] == 'none':
            img = cv2.imread(pathReadFrames + '/'+detect['name'])
            resized_image = cv2.resize(img, (224, 224)) 
            cv2.imwrite(pathSaveFrames + '/'+detect['name'], resized_image)
            
        elif len(detect['result']) == 1:
            for bbox in detect['result']:
                img = cv2.imread(pathReadFrames + '/'+detect['name'])
                cropped_image = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                resized_image = cv2.resize(cropped_image, (224, 224)) 
                cv2.imwrite(pathSaveFrames + '/'+detect['name'], resized_image)
        else:
            pixelsave = []
            for bbox in detect['result']:
                pixelNum = (int(bbox[2]) - int(bbox[0])) + (int(bbox[3]) - int(bbox[0]))
                pixelsave.append(pixelNum) 

            #get index of max value:
            ind = np.argmax(pixelsave)
                            
            img = cv2.imread(pathReadFrames + '/'+detect['name'])
            cropped_image = img[int(detect['result'][ind][1]):int(detect['result'][ind][3]), int(detect['result'][ind][0]):int(detect['result'][ind][2])]
            resized_image = cv2.resize(cropped_image, (224, 224)) 
            cv2.imwrite(pathSaveFrames + '/'+detect['name'], resized_image)

def drawBbxFacesRetinaCUT(res, pathReadFrames, pathSaveFrames):
    for detect in res:
        if detect['result'][0] == 0:
            img = cv2.imread(pathReadFrames + '/'+detect['name'])
            resized_image = cv2.resize(img, (224, 224)) 
            cv2.imwrite(pathSaveFrames + '/'+detect['name'], img)
        elif len(detect['result']) == 1:
            img = cv2.imread(pathReadFrames + '/'+detect['name'])
            cropped_image = img[detect['result'][0][1]:detect['result'][0][3], detect['result'][0][0]:detect['result'][0][2]]
            resized_image = cv2.resize(cropped_image, (224, 224)) 
            cv2.imwrite(pathSaveFrames + '/'+detect['name'], resized_image)
        else:
            pixelsave = []
            for bbox in detect['result']:
                pixelNum = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                pixelsave.append(pixelNum) 
            
            #get index of max value:
            ind = np.argmax(pixelsave)
            
            img = cv2.imread(pathReadFrames + '/'+detect['name'])
            cropped_image = img[int(detect['result'][ind][1]):int(detect['result'][ind][3]), int(detect['result'][ind][0]):int(detect['result'][ind][2])]
            resized_image = cv2.resize(cropped_image, (224, 224)) 
            cv2.imwrite(pathSaveFrames + '/'+detect['name'], resized_image)

def personFaceExtraction(pathFile=r'./ChaLearn_First_Impression/train/train/train_frames_all', framesSavePerson=r'./ChaLearn_First_Impression/train/train/train_person_frames', framesSaveFace=r'./ChaLearn_First_Impression/train/train/train_face_frames', filePathPersonFaceSave=r'./ChaLearn_First_Impression/train/train/'):
    personRes = personPrediction(pathFile+'/', filePathPersonFaceSave)
    faceRes = facePrediction(pathFile +'/', filePathPersonFaceSave)

    #personRes = openFilePkl(r'./ChaLearn_First_Impression/val/val/personBbx.pickle')
    #faceRes = openFilePkl(r'./ChaLearn_First_Impression/val/val/faceBbx.pickle')

    drawBbxWithCutPerson(personRes,pathFile, framesSavePerson)
    drawBbxFacesRetinaCUT(faceRes, pathFile, framesSaveFace)

    #val current
    
if __name__ == '__main__':
    personFaceExtraction()


    
    
    