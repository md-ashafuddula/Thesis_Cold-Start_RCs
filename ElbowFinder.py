# clustering dataset
# determine k using elbow method
 
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import csv 
import time

def elbowFind(init,final):
    
    start_time = time.time()
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_trainSet.csv"
    
    file = open(path,newline='')
    
    read = csv.reader(file)
    trainSet = [] #empty list
    for row in read:              #column1 is user_id 
        feature_1 = float(row[1]) #converting column 2 to float
        feature_2 = float(row[2]) #converting column 3 to float
        trainSet.append([feature_1,feature_2]) #append to column into data array
    file.close()
    #print(trainSet[0],'\n',trainSet[942],' len: ',len(trainSet))
    x1 = np.array(trainSet)
    
    #plt.plot()
    #plt.title('Dataset')
    #plt.scatter(x1[:,0], x1[:,1])
    #plt.show()
    # 
    # create new plot and data
    plt.plot()
    X = x1
     
    # create new plot and data
    plt.plot()
    X = np.array(X)
    #print(X)
    k_Len = len(X)
    k_last = int(k_Len/20)
    
    print('Train Class Len : ',k_Len)
    print('cluster Length: ',k_last)
    
    colors = ['b', 'g', 'r']
    markers = ['o', 'v', 's']
     
    # k means determine k
    distortions = []
    K = range(init,final) #93 #71 #49
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
     
    # Plot the elbow
    
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    print('Used Time',time.time() - start_time,'s')
    return('Elbow Success !!!')