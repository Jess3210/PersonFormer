import pickle
import matplotlib.pyplot as plt

#-----------------Save and Plot the results of Training ------------------
#open pickle var
def openFilePkl(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f, encoding="latin1") 
    return loaded

#save pickle var
def saveFilePkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

#save results as dictionary      
def saveResults(dic, name):
        f = open(name,"w")
        f.write(str(dic))
        f.close()

#plot Results
def plotResults(valueTrain, valueVal, epoch_num, loss=True, pathSave='./visionAudioTextM2TransformerVisionAudio'):
        #plt.figure(figsize=(34, 15), dpi=200)
        lossSave = pathSave + '/loss_graph.png'
        accuracySave = pathSave + '/accuracy_graph.png'
        
        plt.plot(range(epoch_num), valueTrain, label='Train')
        plt.plot(range(epoch_num), valueVal, label='Validation')

        if loss == True:
            #Plot Loss
            plt.title('Loss per Epoch')
            plt.ylabel('Loss')
        else:
            #Plot Accuracy
            plt.title('Accuracy per Epoch')
            plt.ylabel('Accuracy')
        
        plt.xlabel('Epoch')
        plt.legend()
        if loss == True:
            plt.savefig(lossSave)
        else:
            plt.savefig(accuracySave)
        plt.show()    
        plt.close()