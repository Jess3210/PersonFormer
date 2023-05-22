import torch
import torch.nn as nn

#-----------------Evaluation metrics / Loss function------------------
#Average Accuracy per Trait
def traitsAverageAccuracy(prediction, groundtruth):
    avgAccPerSample = 1 - torch.abs(torch.subtract(groundtruth, prediction))
    if (len(prediction) != 5):
        sumAccPerBatch = torch.sum(avgAccPerSample, axis = 0) / len(prediction)
    else:
        sumAccPerBatch = avgAccPerSample
    return sumAccPerBatch

#Mean Average Accuracy over all traits
def meanAverageAccuracy(traitsAvgAccuracy, lenTraits):
    if (len(traitsAvgAccuracy) != 5):
        meanAvgAcc = torch.sum(traitsAvgAccuracy, axis= 1) / lenTraits
    else:
        meanAvgAcc = torch.sum(traitsAvgAccuracy) / lenTraits
    return meanAvgAcc

#MSE per Trait
def traitsMSE(prediction, groundtruth):
    criterionPerClass = nn.MSELoss(reduction = 'none')
    lossPerClass = criterionPerClass(prediction, groundtruth)
    if (len(prediction) != 5):
        lossPerBatch = torch.sum(lossPerClass, axis = 0) / len(prediction)
    else:
        lossPerBatch = lossPerClass
    return lossPerBatch