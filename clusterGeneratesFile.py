# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 04:06:01 2017

@author: Nezamul Islam A R
"""

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np 
from sklearn.cluster import KMeans
import csv
import time
import random

def clusterGen(clsNo):
    
    start_time = time.time()
    #"H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/building Files/test.csv"
    #==================================================== Feature File Input
    #Implementation\Final KNN Implementation\2\Data_R
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_userFeatureWithId.csv"
    #2_userFeatureWithId  k = 47,63,67,71,85
    #1_testFeature50 k = 3
    #FeatureOld
    
    file = open(path,newline='')
    
    read = csv.reader(file)
    data = [] #empty list
    data1 = []
    for row in read:
        feature_1 = int(row[0]) #converting column 1 to int [ID]
        feature_2 = float(row[1]) #converting column 2 to float
        feature_3 = float(row[2])
        data.append([feature_1,feature_2,feature_3]) #append to column into data array
        data1.append([feature_2,feature_3])
    file.close()
    
    #====================================================
    #for i in range (len(data)):
    #   print(data[i]) #Now They are all float
    
    #print(X)
    
    #==================================================== KMeans Clustering
    defineClass = clsNo #defined number of class23,31,39
    #need to use root(n/2) = 55
    lenData = len(data); #set length of features data
    calculateTest = int(lenData/4) #randomly taken 1/5 partition of setData
    #noTest = 20 #20 random data will be taken in testSet
    
    clf = KMeans(n_clusters = defineClass) 
    clf.fit(data1) #set Data for clustering
    
    centroids = clf.cluster_centers_  #set centroid
    
    labels = clf.labels_ #reserved Class
    #means color may be 7 diffent
    #may be single or double coutetion
    colors = 40*['g.',"r.","c.","b.","k.","y.",'m.'] 
    
    #====================================================RandomStart poin
    randTest = int(random.uniform(0,lenData-calculateTest)) #generates random numbers as integer
    randTest = 0
    print("No. starts in TestDataSet -> ",randTest) #print randomNumber
    #randTest = 8
    #====================================================Classifing -> TestDataSetBuilding
    classFile ="H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_trainSet.csv"
    fileTest = open(classFile,'w')
    j = 0
    for i in range (len(data) - calculateTest):
        #print(j,"->")
        if(j == randTest):
            j += calculateTest
           # print(j)
        trainStr = str(data[j][0])+','+str(data[j][1])+','+str(data[j][2])+'\n'
        fileTest.write(trainStr)
        j += 1
    fileTest.close()
    #====================================================LActualLabesls Out Into File
    classFile = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_ALL_ActualStoringClusteringClass.csv"
    file22 = open(classFile,'w')
    #write = csv.writer(file22)
    #====================================================Modified Labels in csv
    #Generated Class writing into file
    lenLabels = len(labels) #set length of class data
    for i in range (lenLabels):
        classVar = str(data[i][0])+','+str(labels[i])+"\n"
        file22.write(classVar)
    file22.close() #class printing file closed
    #====================================================Labels Out in File
    #Printing all the labels to file
    classFile = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_TrainClusteringClass.csv"
    file2 = open(classFile,'w')
    #write = csv.writer(file2)
    #====================================================Modified Labels in csv
    #Generated Class writing into file
    j = 0
    for i in range (0,lenLabels-calculateTest):
        if(j == randTest):
            j += calculateTest
            #print(j,"->Cond : ",randTest)
        classVar = str(data[j][0])+','+str(labels[j])+",\n"
        #print(j)
        j += 1
        file2.write(classVar)
    file2.close() #class printing file closed
    #====================================================Retrieve Cluster Class
    path3 = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_TrainClusteringClass.csv"
    file3 = open(path3,newline='')
    
    read = csv.reader(file3)
    labelC = [] #empty list
    for row in read:
        feature_0 = float(row[0])
        feature_1 = float(row[1]) #converting column 1 to float
        labelC.append([feature_0,feature_1]) #append to column into data array
        #print(row)
    file3.close()
    
    #for i in range (len(labelC)):
    #    print(int(labelC[i][0])," ",int(labelC[i][1]))
    #====================================================Traindataset    
    train = []
    j = 0
    for i in range (len(data) - calculateTest):
        if(j == randTest):
            j += calculateTest
        train.append([data[j][1],data[j][2]])
        j += 1
    #====================================================Testdataset
                         
    testSet = [] #initial testSet
    
    for i in range (randTest,randTest+calculateTest):
        f1 = data[i][1] 
        f2 = data[i][2]
        testSet.append([f1,f2])
      
    #====================================================Labels Identify & trainingFitMatrix(i = defineClass X n = maxClass)
    
    maxClass = 0;
    particularClassFind = [] #empty List
    
    for i in range (defineClass):
        particularClassFind.append(0) #initializing classfind 1D List
    
    #Set value in classFind list
    lenLabels = len(labelC)
    #print(labelC)
    for i in range (lenLabels):
        particularClassFind[int(labelC[i][1])] += 1
    #=========2d trainingFitMatrix
    """
    Class_ 0  -> Total Member :  14
    Class_ 1  -> Total Member :  11
    Class_ 2  -> Total Member :  9
    """
    print("Class no. -> ",defineClass) 
    maxClass = max(particularClassFind) 
    print('Max Class is = ',maxClass)              
    for i in range (defineClass):
        print('Class_',i,' -> Total Member : ',particularClassFind[i])
    
    #for i in range(lenLabels):
    #    print(labelC[i],'bul')
    print(lenLabels)    
    r,c = defineClass,maxClass
    trainingFitMatrix = [[0 for z in range(c)] for y in range(r)]
    k = 0
    countTmf = 0
    for i in range (0,r):
        for j in range (0,lenLabels):     
            if(labelC[j][1] == i):
                trainingFitMatrix[i][k] = int(labelC[j][0])
                k += 1
                countTmf += 1
            #print(labelC[j][0],' ',j,end = "\n")                         
        k = 0         
        
    #====================================================print in CSV all trainingFitMatrix
    #trainMatrixShape
    path0 = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_shapeFit.csv"
    file0 = open(path0,'w')
    writer = csv.writer(file0)
    writer.writerow([defineClass,maxClass,countTmf])
    file0.close()
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_trainingFitMatrix.csv"
    file = open(path,'w')
    
    for i in range(r):
        rn = ""
        for j in range(c):
            rn += str(trainingFitMatrix[i][j])
            if(j < c-1):
                rn += ','
        rn += '\n'
        file.write(rn)    
        
    file.close()
    #====================================================Print TestDataset into file
    #Printing all the labels to file
    testDataFile = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_testdataSet.csv"
    testLabelsFile = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_testLabelsSet.csv"
    file3 = open(testDataFile,'w')
    fileLabel = open(testLabelsFile,'w')
    #write = csv.writer(file2)
    
    #Generated Class writing into file
    #lenTestData = len(testSet) #set length of class data
    j = randTest #24
    
    for i in range (0,calculateTest):
        f0 = data[j][0]
        f1 = data[j][1]
        f2 = data[j][2]
        testClass = str(f0)+","+str(labels[j])+'\n'
        j += 1
        fileLabel.write(testClass)
        featuresTestData = str(f0)+","+str(f1)+","+str(f2)+"\n"
        file3.write(featuresTestData)
        
    file3.close() #class printing file closed
    fileLabel.close() #Test Label close
    print("train File : ",len(train)," test File : ",len(testSet),' total Data : ',lenData," actualFile : ",lenData)
    #printing total  time
    #====================================================KMeans cluster Ended
    
    #for i in range(lenData):
    #    plt.plot(data[i][1],data[i][2],colors[labels[i]],markersize = 3)
    #    
    #plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',s = 60,linewidths = 100)
    #for i in range (len(testSet)):
    #    plt.scatter(testSet[i][0],testSet[i][1],marker = 'p',s = 3,linewidths = 10) #addin new scatter in same plot
    ##plt.xlim([-.1, .3])
    ##plt.ylim([-.3, .35])
    #plt.show()
    
    #====================================================Centorids file
    
    centroidFile = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_centroidSet.csv"
    file = open(centroidFile,'w')
    for i in range (0,defineClass):
        c1 = centroids[i,0]
        c2 = centroids[i,1]
        rowM = str(c1)+','+str(c2)+'\n'
        file.write(rowM)
    file.close()
    
    print('Used Time',time.time() - start_time,'s')
    return('Culster Success .... !!!')