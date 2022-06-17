# Import Libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#cutting image into images
def imaged_grid(img , row , col ):

    x , y= img.shape 
    assert x % row == 0, x % row .format(x, row)
    assert y % col == 0, y % col.format(y, col)
    
    
    return (img.reshape ( x //row, row, -1, col)
               .swapaxes(1,2)
               .reshape(-1, row, col))

def get_centroid(img):
 
    feature_vector = []
 
    for grid in imaged_grid(img , 2 , 2 ) :
        
        X_center = 0 
        Y_center = 0 
        sum = 0

        for x in range(2):
          for y in range(2):
            X_center=X_center+x*grid[x][y]
            Y_center=Y_center+y*grid[x][y]
            sum+=grid[x][y]

        if sum == 0 :
            feature_vector.append(0)
            feature_vector.append(0)
        
        else :
          feature_vector.append( X_center/ sum )
          feature_vector.append(Y_center/ sum )
     
    return np.array(feature_vector)

#classify featuers by KNN from scratch

class KNN:
    distances = [[]*2]
    final_label = []
    def getDistance(self,test_vector,train_feature_vectors,train_labels): 
        for i in range(len(train_feature_vectors)):
            distance = np.linalg.norm(test_vector-train_feature_vectors[i]) #calculate the distance between test vector and train vectors
            self.distances.append([distance,train_labels[i]]) #append the distance and label to the distances array
        self.distances = sorted(self.distances, key=lambda x:x[0]) #sort the distances array by the distance
        return self.distances
    def getLabel(self,k):
        labels = []
        for i in range(k):
            labels.append(self.distances[i][1]) 
        return labels
    def getNearestNeighbor(self,k):
        labels = self.getLabel(k)
        return max(set(labels), key=labels.count)
    def Classifier(self,k,train_features, test_features, Trainlabels):
        for i in range(len(test_features)):
            self.distances = []
            self.getDistance(test_features[i],train_features,Trainlabels) #fill distances list with distances and labels 
            self.final_label.append(self.getNearestNeighbor(k)) #get the label of the nearest k neighbor
        return self.final_label
    
#__________________Main_______________

#loading
(Datatrain,Trainlabels), (Datateast,Teastlabels) = mnist.load_data()
 
#Take some of data
Datatrain=Datatrain[0:1000]
Trainlabels=Trainlabels[0:1000]
Datateast=Datateast[0:100]
Teastlabels=Teastlabels[0:100]

#shape of dataset
self = KNN()
train_features = [get_centroid(img)  for img in Datatrain  ]
test_features = [get_centroid(img)  for img in Datateast ]
KNN_Prediction=KNN.Classifier(self,3,train_features, test_features, Trainlabels)

wrong_classifier =0
for i in range(len(Teastlabels)):
  if KNN_Prediction[i] != Teastlabels[i]:
    wrong_classifier+=1
accuracy = (100-(wrong_classifier/len(Teastlabels))*100)
print("Scratch Accuracy", accuracy , "%")    



