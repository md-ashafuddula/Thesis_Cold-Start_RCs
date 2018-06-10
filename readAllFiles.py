# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 05:26:19 2017
    
@author: Nezamul Islam A R
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import csv
import time
import numpy as np 
import operator
from sklearn import tree
#from sklearn import svm
#import math

def readFiles(clsDef):     
    
    start_time = time.time()
    
    defineClass = clsDef #defined number of class
    #reducedDimNew  k = 47
    #test k = 3
    #FeatureOld
    #==================================================== Train File Input
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_trainSet.csv"
    
    file = open(path,newline='')
    
    read = csv.reader(file)
    trainSet = [] #empty list
    trainSet1 = [] #empty useinglist
    for row in read:
        feature_0 = int(row[0])
        feature_1 = float(row[1]) #converting column 1 to float
        feature_2 = float(row[2]) #converting column 2 to float
        trainSet.append([feature_0,feature_1,feature_2]) #append to column into data array
        trainSet1.append([feature_1,feature_2])
    file.close()
    #print(trainSet1," len ",len(trainSet1))
    #==================================================== Train Label File Input
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_TrainClusteringClass.csv"
    
    file = open(path,newline='')
    
    read = csv.reader(file)
    trainLabel = [] #empty list
    trainLabel1 = [] #empty usinglist
    for row in read:
        label_0 = int(row[0])
        label_1 = float(row[1]) #converting column 1 to float
        trainLabel.append([label_0,label_1]) #append to column into data array
        trainLabel1.append(label_1)
        
    file.close()
    #print(trainLabel1," len ",len(trainLabel1))
    #==================================================== test File Input
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_testdataSet.csv"
    
    file = open(path,newline='')
    
    read = csv.reader(file)
    testSet = [] #empty list
    testSet1 = [] #empty usinglist
    for row in read:
        feature_0 = int(row[0])
        feature_1 = float(row[1]) #converting column 1 to float
        feature_2 = float(row[2]) #converting column 2 to float
        testSet.append([feature_0,feature_1,feature_2]) #append to column into data array
        testSet1.append([feature_1,feature_2])
        
    file.close()
    #print(testSet1," len ",len(testSet1))
    #==================================================== test Label File Input
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_testLabelsSet.csv"
    
    file = open(path,newline='')
    
    read = csv.reader(file)
    testLabel = [] #empty list
    testLabel1 = [] #empty list
    for row in read:
        label_0 = int(row[0])
        label_1 = float(row[1]) #converting column 1 to float
        testLabel.append([label_0,label_1]) #append to column into data array
        testLabel1.append(label_1) #append to column into data array
    file.close()
    #print(testLabel," len ",len(testLabel))
    #==================================================== centroid Input
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_centroidSet.csv"
    
    file = open(path,newline='')
    
    read = csv.reader(file)
    centroidSet = [] #empty list
    for row in read:
        c_1 = float(row[0]) #converting column 1 to float
        c_2 = float(row[1]) #converting column 2 to float
        centroidSet.append([c_1,c_2]) #append to column into data array
    file.close()
    #print(centroidSet," len ",len(centroidSet))
    #====================================================Data must same as KMeans generating file
    noTest = int(len(testSet))
    #need to use root(n/2) = 55
    #lenData = len(data); #set length of features data
    #==================================================== Plot Data
    """
    colors = 10*['g.',"r.","c.","b.","k.","y.",'m.'] 
    for i in range(len(trainSet1)):
        plt.plot(trainSet1[i][0],trainSet1[i][1],marker = '*',markersize = 3)
        
    for i in range (len(centroidSet)):    
        plt.scatter(centroidSet[i][0],centroidSet[i][1],marker = 'x',s = 60,linewidths = 100)
    #print(centroidSet)    
    for i in range (len(testSet1)):
        plt.scatter(testSet1[i][0],testSet1[i][1],marker = 'p',s = 3,linewidths = 10) #addin new scatter in same plot
    
    plt.show()
    """
    #file.close()
    
    print('trnSet',len(trainSet1),' trnLabel: ',len(trainLabel1),' tstSet: ',len(testSet1),' tstLabel: ',len(testLabel1),' centroid: ',len(centroidSet))
    #print('\nHi ther, ',trainSet1)
    #=======2d modification  trainSet & trainLabel
    #=======2d modification  testSet & testLabel
    
    #==================================================== Implementation KNN
    
    def createDataSet():
    
        group = np.array(trainSet1)
        labelsClass = trainLabel1
        return group, labelsClass
    
    def calcDistance(inX, dataSet, labels, k):
        # shape is a tuple that gives dimensions of the array
        # shape[0] returns the number of rows, here will return 4
        dataSetSize = dataSet.shape[0]  # dataSetSize = 4
    
        # numpy.tile(A, reps): construct an array by repeating A the number of times given by reps
        # if reps has length d, the result will have dimension of max(d, A.ndim)
        # tile(inX, (dataSetSize,1)) will return [ [0,0] [0,0] [0,0] [0,0] ]
        # diffMat is [ [1, 1], [1, -1], [-2, 2], [-2, 1] ]
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    
        # **2 means square
        sqDiffMat = diffMat ** 2
    
        # sqDistances = x^2 + y^2
        sqDistances = sqDiffMat.sum(axis=1)
        # distance is equal to the square root of the sum of the squares of the coordinates
        # distance = [2, 2, 8, 5]
        distances = sqDistances ** 0.5
    
        # numpy.argsort() returns the indices that would sort an array
        # here returns [0 1 3 2]
        sortedDistIndices = distances.argsort()
        return sortedDistIndices
    
    def findMajorityClass(inX, dataSet, labels, k, sortedDistIndices):
        classCount = {}
    
        # iterate k times from the closest item
        for i in range(k):
            voteIlabel = labels[sortedDistIndices[i]]
            # increase +1 on the selected label
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
        # classCount dictionary : {'Action': 2, 'Romantic': 1}
        # sort ClassCount Descending order
    
        return sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    
    
    def classify0(inX, dataSet, labels, k):
        # calculate the distance between inX and the current point
        sortedDistIndices = calcDistance(inX, dataSet, labels, k)
        # take k items with lowest distances to inX and find the majority class among k items
        sortedClassCount = findMajorityClass(inX, dataSet, labels, k, sortedDistIndices)
        # sortedClassCount is now [('Action', 2)], therefore return Action
        return sortedClassCount[0][0]
    
    
    group, labelsClass = createDataSet()
    
    check = 0 #no of correct classified data
    RMSE = 0
    newUserClassKNN = []    
    for j in range(len(testSet)):
            result = classify0(testSet1[j], group, labelsClass,defineClass) 
            #group = training Dataset
            #labels = class serially as training dataset {particularClassFind}
            #k = 3 small Odd numbers   
            #no. of correct classification / defineClass = accuracy
            resultClass = int(result)
            actualClass = int(testLabel1[j])
            if((resultClass - actualClass) == 0):
                check += 1
                #print(check)
            newUserClassKNN.append(resultClass)
            #print(testSet[j][0]," KNN -> ",resultClass," Actual -> ",actualClass)  
    #==================================== class
    #print(newUserClassKNN)
    classNumber = []
    cN = 0;
    for i in range (0,defineClass):
        for j in range(0,len(newUserClassKNN)):
            if(int(newUserClassKNN[j]) == i):
                cN += 1
        classNumber.append(cN)
        cN = 0
    mxknclass = max(classNumber)
    print('Max member : ',mxknclass,'Class members : ',classNumber)     
    #Accuracy
    classifyingAccuracy = (check/noTest)*100
    print('Correct : ',check," KNN Accuracy is : ",classifyingAccuracy,"%")
    #==================================== newUserKnnMatrinx  
    r,c = defineClass,mxknclass
    newUserKnnMatrinx   = [[0 for i in range(c)]for j in range(r)]
    flagC = 0
    for i in range (0,r):
        for j in range (0,len(newUserClassKNN)):
            if(newUserClassKNN[j] == i):
                newUserKnnMatrinx[i][flagC] = testSet[j][0]
                flagC += 1
        flagC = 0
        
    #for i in range (0,r):
    #    for j in range (0,c):
    #        print(newUserKnnMatrinx[i][j],end = ' ')
    #    print('\n')  
    #==================================================== Implementation DT
    group  = np.array(trainSet1)
    labelsClass = trainLabel1
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(group,labelsClass)
    resultDT = clf.predict(testSet1) #Class prediction
    chekDTAcc = 0
    newUserClassDT = [] 
    
    for i in range (len(testSet1)):
        resultClass = int(resultDT[i])
        actualClass = int(testLabel1[i])
        if((resultClass - actualClass) == 0):
                chekDTAcc += 1
        #print(testSet[i][0]," DT -> ",resultClass," Actual -> ",actualClass)
        newUserClassDT.append(resultClass) 
    classifyingAccuracy = (chekDTAcc/noTest)*100                
    #==================================== class
    #print(newUserClassDT)
    
    classNumberDT = []
    cN = 0;
    for i in range (0,defineClass):
        for j in range(0,len(newUserClassDT)):
            if(int(newUserClassDT[j]) == i):
                cN += 1
        classNumberDT.append(cN)
        cN = 0
    mxDTclass = max(classNumberDT)
    print('Max member : ',mxDTclass,'Class members : ',classNumberDT) 
    
    #Accuracy                 
    print('Correct : ',chekDTAcc," DT Accuracy is : ",classifyingAccuracy,"%")
    
    
    #==================================== newUserDtMatrinx   
    r,c = defineClass,mxDTclass
    newUserDtMatrinx = [[0 for i in range(c)]for j in range(r)]
    flagC = 0
    for i in range (0,r):
        for j in range (0,len(newUserClassDT)):
            if(newUserClassDT[j] == i):
                newUserDtMatrinx[i][flagC] = testSet[j][0]
                flagC += 1
        flagC = 0
        
    #for i in range (0,r):
    #    for j in range (0,c):
    #        print(newUserDtMatrinx[i][j],end = ' ')
    #    print('\n')    
    
          
    #====================================================file Print newUserClassFitMatrix
    
    #write KNN
    pathKnn = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_newUserFitMatrixKNN.csv"
    fileD = open(pathKnn,'w')
    r = defineClass
    c = mxknclass
    
    for i in range(r):
        rn = ""
        for j in range(c):
            rn += str(newUserKnnMatrinx[i][j])
            if(j < c-1):
                rn += ','
        rn += '\n'
        fileD.write(rn)    
        
    fileD.close()
    
    #Wrte DT
    pathDt = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_newUserFitMatrixDT.csv"
    fileK = open(pathDt,'w')
    r = defineClass
    c = mxDTclass
    
    for i in range(r):
        rn = ""
        for j in range(c):
            rn += str(newUserDtMatrinx[i][j])
            if(j < c-1):
                rn += ','
        rn += '\n'
        fileK.write(rn)    
        
    fileK.close()
    #==================================================== matrixShapeKNN&DT
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_shapeFit.csv"
    
    file = open(path,newline='')
    
    read = csv.reader(file)
    k = 0
    for row in read:
        if(k == 0):
            label_0 = str(row[0])
            label_1 = str(row[1]) 
            k += 1
    file.close()
    
    #trainMatrixShape
    path0 = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_shapeFit_2.csv"
    file0 = open(path0,'w')
    writer = csv.writer(file0)
    knnShape = str(defineClass)+','+str(mxknclass) #Knn Shape
    dtShape = str(defineClass)+','+str(mxDTclass) #DT shape
    mergWr = label_0+','+label_1+'\n'+knnShape+'\n'+dtShape
    file0.write(mergWr)
    
    file0.close()
    #print(mergWr)
    #====================================================fething Training Fit Mat
    tr = int(label_0)
    tc = int(label_1)
    print('TFM : tr = ',tr,' tc = ',tc)
    TFM = [[0 for x in range(tc)] for y in range(tr)]
    class_i = [0 for z in range(tc)]
    #print(class_i)
    path_tr = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_trainingFitMatrix.csv"
    file = open(path_tr,newline='')
    
    read = csv.reader(file)
    k = 0
    for row in read:
        for col in range(0,tc):
            TFM[k][col] = int(row[col])        
        k += 1  
    file.close()
    #print('Training U_ID')
    #for i in range (0,tr):
    #    for j in range (0,tc):
    #        print(TFM[i][j],end = ' ')
    #    print('\n')
    #====================================================SeenMovieByClass_TFM
    #testClass = 5 #defineClass
    #defineClass = testClass
    #tr = testClass
    
    sr = 1682#1682 | 20 #100 #total different movie
    sc = defineClass
    smvc = [[0 for x in range(sc)] for y in range(sr)]
    #Updating smvc for Training U_id class
    path_rate = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_ratings2.csv"
    
    #1_ratings 100000
    #1_testRating 99
    rateR = 100000#100000 | 99 #rating mat Len
    rateC = 3 #rating mat Column
    
    #Rating matrix file into matrix
    rm = [[0 for x in range(rateC)] for y in range(rateR)]
    file = open(path_rate,newline='')
    read = csv.reader(file)
    
    i = 0
    RatValerr = 0
    for row in read:
        #print('rating mat len : ',len(row))
        if(len(row) != 0):
            try:
                rm[i][0] = int(row[0])
                rm[i][1] = int(row[1]) 
                rm[i][2] = int(row[2])
                i += 1
            except :
                RatValerr += 1
    file.close()
    rm_Len = i
    print('rating Mat Len : ',rm_Len,' Error : ',RatValerr)
    #.....Reduce Rating Mat
    
    rm1Len = 19747
    rm2Len = 44317 - rm1Len + 1
    rm3Len = 66277 - 44317 + 1
    rm4Len = 85532 - 66277 + 1
    rm5Len = 100000 - 85532 + 1
    rm1 = [[0 for x in range(rateC)] for y in range(rm1Len)]
    rm2 = [[0 for x in range(rateC)] for y in range(rm2Len)]
    rm3 = [[0 for x in range(rateC)] for y in range(rm3Len)]
    rm4 = [[0 for x in range(rateC)] for y in range(rm4Len)]
    rm5 = [[0 for x in range(rateC)] for y in range(rm5Len)]
    
    for i in range(0,rm1Len):
        for j in range(0,rateC):
            rm1[i][j] = rm[i][j]
    
    k = 0        
    er = 0
    
    for i in range(rm1Len,44317):
        try:
            rm2[k][0] = rm[i][0]
            rm2[k][1] = rm[i][1]
            rm2[k][2] = rm[i][2]
            k += 1
        except:
            er += 1
            print('rm2 ',er)
    
    k = 0        
    er = 0
    for i in range(44317,66277):
        for j in range(0,rateC):
            try:
                rm3[k][j] = rm[i][j]            
            except:
                er += 1
                print('rm3 ',er)
        k += 1
    
    k = 0        
    er = 0
    for i in range(66277,85532):
        for j in range(0,rateC):
            try:
                rm4[k][j] = rm[i][j]
            except:
                er += 1
                print('rm4 ',er)
        k += 1
        
    k = 0
    er = 0
    for i in range(85532,100000):
        for j in range(0,rateC):
            try:
                rm5[k][j] = rm[i][j]
            except:
                er += 1
                print('rm5 ',er)
        k += 1
    
    #===================================================printAll Rating
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ratings_a1.csv"
    file = open(path,'w')
    
    rn = ''
    err = 0
    
    for i in range(0,rm1Len):
        rn = ''
        for j in range(0,rateC):
            try:
                rn += str(rm1[i][j])+','            
            except:
                err += 1
        rn += '\n'
        file.write(rn) 
        
    file.close()
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ratings_a2.csv"
    file = open(path,'w')
    
    rn = ''
    err = 0
    
    for i in range(0,rm2Len):
        rn = ''
        for j in range(0,rateC):
            try:
                rn += str(rm2[i][j])+','            
            except:
                err += 1
        rn += '\n'
        file.write(rn) 
        
    file.close()
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ratings_a3.csv"
    file = open(path,'w')
    
    rn = ''
    err = 0
    
    for i in range(0,rm3Len):
        rn = ''
        for j in range(0,rateC):
            try:
                rn += str(rm3[i][j])+','            
            except:
                err += 1
        rn += '\n'
        file.write(rn) 
        
    file.close()
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ratings_a4.csv"
    file = open(path,'w')
    
    rn = ''
    err = 0
    
    for i in range(0,rm4Len):
        rn = ''
        for j in range(0,rateC):
            try:
                rn += str(rm4[i][j])+','            
            except:
                err += 1
        rn += '\n'
        file.write(rn) 
        
    file.close()
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ratings_a5.csv"
    file = open(path,'w')
    
    rn = ''
    err = 0
    
    for i in range(0,rm5Len):
        rn = ''
        for j in range(0,rateC):
            try:
                rn += str(rm5[i][j])+','            
            except:
                err += 1
        rn += '\n'
        file.write(rn) 
        
    file.close()
    #U_id | mov_id | Rating
    """
    #for i in range (0,rateR):
    #    for j in range (0,rateC):
    #        print(rm[i][j],end = ' ')
    #    print('\n')
    """
    #Updating smvc
    #trainingFit (tr X tc)
    
    testPrint = 0
    for r in range(0,tr):
        for c in range(0,tc):
            oldU = TFM[r][c]
            if(oldU != 0):
                for i in range(0,rateR):
                    flag = 0
                    if(rm[i][0] == oldU):
                        for j in range(0,sr):
                            if((smvc[j][r] != rm[i][1]) and (smvc[j][r] == 0) and (flag == 0)):
                                smvc[j][r] = rm[i][1]
                                print('smvc Up see: ',testPrint)
                                testPrint += 1
                                break
                            if(smvc[j][r] == rm[i][1]):
                                flag = 1
                                break
            else:
                break
            
    flag = 0
    for countV in range(0,sr):
        chk = 0
        if(flag == 0):
            for j in range(0,sc):
                if((smvc[countV][j] == 0) ):
                    chk +=1
        if(chk == sc):
            flag = 1
            break
    print('smvc_K length: ',countV)
    smvc_Len = countV
    #print('SMVC_K').........
    #for i in range (0,smvc_Len):
    #    for j in range (0,sc):
    #        print(smvc[i][j],end = ' ')
    #    print('\n')
    #=============write predicted matrix
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_SMVC.csv"
    file = open(path,'w')
    
    rn = ''
    err = 0
    
    for i in range(0,sr):
        rn = ''
        for j in range(0,sc):
            try:
                rn += str(smvc[i][j])+','            
            except:
                err += 1
        rn += '\n'
        file.write(rn) 
        
    file.close()
    print('smvc error print ',err)
    
    #==================================================== Total Movie in a Class  [TTM]
    
    TTM = [0]*defineClass
    ttm_er = 0
    for i in range(0,defineClass):
        count = 0    
        for j in range(0,smvc_Len):
            try:
                print(smvc[j][i],end = ' ')
                if(smvc[j][i] == 0):
                    break
                else:
                    count += 1
            except:
                ttm_er += 1
        TTM[i] = count
        print('\n')   
        
    print('TTM error print ',ttm_er)            
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_TTM_Matrix.csv"
    file = open(path,'w')
    
    rn = ''
    
    for i in range(0,defineClass):
        rn = str(TTM[i])+'\n'
        file.write(rn) 
    file.close()
    #==================================================== newClassMovie_prediction
    #class | movie_id | Rating
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_predictedMatrix.csv"
    file = open(path,'w')
    
    rn = ''
    errP = 0
    
    predict_list = smvc_Len * defineClass
    pre_C = 5 #class | movie_id | total_Rating | count | avg
    
    pm = [[0 for x in range(pre_C)] for y in range(predict_list)]
    
    testPrint = 0
    c_p = 0
    err = 0
    err1 = 0
    flagSe = 0
    for c in range(0,defineClass): #smvc column for class [0,1,2]
        for r in range(0,smvc_Len): #search in smvc mat (0-28)
            total_rate = 0
            countSeenTimes = 0
            p_len = 0
            if(smvc[r][c] == 0):
                break
            else:
                for i in range(0,tc): #search in TFM mat for avg rating (0-15)
                #TFM[c][i] U_id
                    if(TFM[c][i] == 0):
                        break
                    
                    if(TFM[c][i]  >= 0 and TFM[c][i]  <= 100):
                        ratingMatinitn = 0
                        ratingMatFinal = 11019
                    elif(TFM[c][i]  >= 101 and TFM[c][i] <= 200):
                        ratingMatinitn = 11020
                        ratingMatFinal = 19747
                    elif(TFM[c][i]  >= 201 and TFM[c][i]  <= 300):
                        ratingMatinitn = 19746
                        ratingMatFinal = 31468
                    elif(TFM[c][i]  >= 301 and TFM[c][i]  <= 400):
                        ratingMatinitn = 31468
                        ratingMatFinal = 44317
                    elif(TFM[c][i]  >= 401 and TFM[c][i] <= 500):
                        ratingMatinitn = 44316
                        ratingMatFinal = 56770
                    elif(TFM[c][i]  >= 501 and TFM[c][i]  <= 600):
                        ratingMatinitn = 56771
                        ratingMatFinal = 66277
                    elif(TFM[c][i]  >= 601 and TFM[c][i]  <= 700):
                        ratingMatinitn = 66276
                        ratingMatFinal = 76420
                    elif(TFM[c][i]  >= 701 and TFM[c][i]  <= 800):
                        ratingMatinitn = 76420
                        ratingMatFinal = 85532
                    elif(TFM[c][i]  >= 801 and TFM[c][i] <= 900):
                        ratingMatinitn = 85532
                        ratingMatFinal = 96103
                    elif(TFM[c][i]  >= 901 and TFM[c][i]  <= 1000):
                        ratingMatinitn = 96103
                        ratingMatFinal = 100000                
                    
                    for j in range(ratingMatinitn,ratingMatFinal): #search in rating for U_id & mvie_Id match (0 -100)
                        flagSe += 1
                        #print('pm flag see: ',flagSe)
                        if((TFM[c][i] < rm[j][0])):
                            break
                        if((TFM[c][i] == rm[j][0]) and (smvc[r][c] == rm[j][1])):
                            try:
                                countSeenTimes += 1
                                total_rate += rm[j][2]
                                avg = total_rate / countSeenTimes
                                break
                            except:
                                err += 1        
            if((total_rate != 0) and (countSeenTimes != 0) and smvc[r][c] != 0 and c != 0 and countSeenTimes >= 2 and avg >= 3.0):
                try:
                    pm[c_p][0] = c
                    pm[c_p][1] = smvc[r][c]
                    pm[c_p][2] = total_rate
                    pm[c_p][3] = countSeenTimes        
                    pm[c_p][4] = avg
                    #=============write predicted matrix
                    try:
                        rn = str(pm[c_p][0])+','+str(pm[c_p][1])+','+str(pm[c_p][2])+','+str(pm[c_p][3])+','+str(pm[c_p][4])+','+'\n'
                        file.write(rn) 
                    except:
                        errP += 1
                        print('pm error ',errP)
                        
                    c_p += 1
                    print('pm Up see: ',testPrint,' class TFM : ',c)  
                    testPrint += 1
                except:
                    err1 += 1
            
        
    print('Predict Matrix Len: ',c_p,' err ',err,' err1 ',err1)
    #for i in range(0,c_p):
    #    pm[i][4] = pm[i][2] / pm[i][3]
    #for i in range(0,c_p):
    #    print(pm[i])
    file.close()
    print('pm error print ',errP)
    
    #==================================================== ErrMat_newUserDT
    r = defineClass #user Row
    c = mxDTclass #user clmn
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ErrMat_dt.csv"
    file = open(path,'w')
    
    err = 0
    
    uR = smvc_Len * tc
    uC = 7
    ErrMat = [[0 for x in range(uC)] for y in range(uR)] 
    
    er = 0
    er1 = 0
    er2 = 0
    er3 = 0
    er4 = 0
    rn = ''
    #define Var
    u_id = 0
    u_class = 0
    m_id = 0
    avgr = 0
    rate= 0 
    avgr = 0
    
    
    x = 0
    for i in range(0,r):
        for j in range(0,c):
            if(newUserDtMatrinx[i][j] == 0):
                break
            else:
                try:
                    for k in range(0,smvc_Len):
                        try:
                            if(smvc[k][i] == 0):
                                break
                            else:
                                u_id = newUserDtMatrinx[i][j]
                                m_id = smvc[k][i]
                                u_class = i
                                
                                if(u_id  >= 0 and u_id  <= 100):
                                    ratingMatinitn = 0
                                    ratingMatFinal = 11019
                                elif(u_id  >= 101 and u_id  <= 200):
                                    ratingMatinitn = 11020
                                    ratingMatFinal = 19747
                                elif(u_id  >= 201 and u_id  <= 300):
                                    ratingMatinitn = 19746
                                    ratingMatFinal = 31468
                                elif(u_id  >= 301 and u_id  <= 400):
                                    ratingMatinitn = 31468
                                    ratingMatFinal = 44317
                                elif(u_id  >= 401 and u_id  <= 500):
                                    ratingMatinitn = 44316
                                    ratingMatFinal = 56770
                                elif(u_id  >= 501 and u_id  <= 600):
                                    ratingMatinitn = 56771
                                    ratingMatFinal = 66277
                                elif(u_id  >= 601 and u_id  <= 700):
                                    ratingMatinitn = 66276
                                    ratingMatFinal = 76420
                                elif(u_id  >= 701 and u_id  <= 800):
                                    ratingMatinitn = 76420
                                    ratingMatFinal = 85532
                                elif(u_id  >= 801 and u_id  <= 900):
                                    ratingMatinitn = 85532
                                    ratingMatFinal = 96103
                                elif(u_id  >= 901 and u_id  <= 1000):
                                    ratingMatinitn = 96103
                                    ratingMatFinal = 100000
                                    
                                for l in range(ratingMatinitn,ratingMatFinal):
                                    try:
                                        if((rm[l][0] == u_id) and (rm[l][1] == m_id)):
                                            rate = rm[l][2]
                                            break
                                    except:
                                        er2 += 1
                                        print('err2 :',er2)
                                flag = 0
                                for m in range(0,predict_list):
                                    if((pm[m][0] == u_class) and (pm[m][1] == m_id) and (flag == 0)):
                                        avgr = pm[m][4]
                                        flag = 1                            
                        except:
                            er1 += 1
                            print('err1 :',er1)              
                                      
                    try:
                        ErrMat[x][0] = u_id
                        ErrMat[x][1] = u_class
                        ErrMat[x][2] = m_id
                        ErrMat[x][3] = avgr
                        ErrMat[x][4] = rate
                        ErrMat[x][5] = (avgr - rate)**2
                        ErrMat[x][6] = (rate)**2
                        print(ErrMat[x][0],' ',ErrMat[x][1],' ',ErrMat[x][2],' ',ErrMat[x][3],' ',ErrMat[x][4],' ',ErrMat[x][5],' ',ErrMat[x][6])
                        
                        rn = str(ErrMat[x][0])+','+str(ErrMat[x][1])+','+str(ErrMat[x][2])+','+str(ErrMat[x][3])+','+str(ErrMat[x][4])+','+str(ErrMat[x][5])+','+str(ErrMat[x][6])+'\n'
                        file.write(rn) 
                        x += 1
                    except:
                        er4 += 1
                        print('err4 :',er4)
                except:
                    er3 += 1
                    print('err3 :',er3)
    
    file.close()
    print('len ErMat ',len(ErrMat),' err1 : ',er1,' err2 : ',er2,' err3 : ',er3,' err4 : ',er4)
    #==================================================== ErrMat_newUserKNN
    r = defineClass #user Row
    c = mxknclass #user clmn
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ErrMat_kn.csv"
    file = open(path,'w')
    
    err = 0
    
    uR = smvc_Len * tc
    uC = 7
    ErrMat_kn = [[0 for x in range(uC)] for y in range(uR)] 
    
    #define Var
    u_id = 0
    u_class = 0
    m_id = 0
    avgr = 0
    rate= 0 
    avgr = 0
    
    
    er = 0
    er1 = 0
    er2 = 0
    er3 = 0
    er4 = 0
    
    x = 0
    for i in range(0,r):
        for j in range(0,c):
            if(newUserKnnMatrinx[i][j] == 0):
                break
            else:
                try:
                    for k in range(0,smvc_Len):
                     
                        try:
                            if(smvc[k][i] == 0):
                                break
                            else:
                                u_id = newUserKnnMatrinx[i][j]
                                m_id = smvc[k][i]
                                u_class = i
                                
                                if(u_id  >= 0 and u_id  <= 100):
                                    ratingMatinitn = 0
                                    ratingMatFinal = 11019
                                elif(u_id  >= 101 and u_id  <= 200):
                                    ratingMatinitn = 11020
                                    ratingMatFinal = 19747
                                elif(u_id  >= 201 and u_id  <= 300):
                                    ratingMatinitn = 19746
                                    ratingMatFinal = 31468
                                elif(u_id  >= 301 and u_id  <= 400):
                                    ratingMatinitn = 31468
                                    ratingMatFinal = 44317
                                elif(u_id  >= 401 and u_id  <= 500):
                                    ratingMatinitn = 44316
                                    ratingMatFinal = 56770
                                elif(u_id  >= 501 and u_id  <= 600):
                                    ratingMatinitn = 56771
                                    ratingMatFinal = 66277
                                elif(u_id  >= 601 and u_id  <= 700):
                                    ratingMatinitn = 66276
                                    ratingMatFinal = 76420
                                elif(u_id  >= 701 and u_id  <= 800):
                                    ratingMatinitn = 76420
                                    ratingMatFinal = 85532
                                elif(u_id  >= 801 and u_id  <= 900):
                                    ratingMatinitn = 85532
                                    ratingMatFinal = 96103
                                elif(u_id  >= 901 and u_id  <= 1000):
                                    ratingMatinitn = 96103
                                    ratingMatFinal = 100000
                                
                                for l in range(ratingMatinitn,ratingMatFinal):
                                    try:
                                        if((rm[l][0] == u_id) and (rm[l][1] == m_id)):
                                            rate = rm[l][2]
                                            break
                                    except:
                                        er2 += 1
                                        print('knerr2 :',er2)
                                flag = 0
                                for m in range(0,predict_list):
                                    if((pm[m][0] == u_class) and (pm[m][1] == m_id) and (flag == 0)):
                                        avgr = pm[m][4]
                                        flag = 1                            
                        except:
                            er1 += 1
                            print('knerr1 :',er1)
                    
                    ErrMat_kn[x][0] = u_id
                    ErrMat_kn[x][1] = u_class      
                    ErrMat_kn[x][2] = m_id
                    ErrMat_kn[x][3] = avgr
                    ErrMat_kn[x][4] = rate
                    ErrMat_kn[x][5] = (avgr - rate)**2
                    ErrMat_kn[x][6] = (rate)**2                       
                    print(ErrMat[x][0],' ',ErrMat[x][1],' ',ErrMat[x][2],' ',ErrMat[x][3],' ',ErrMat[x][4],' ',ErrMat[x][5],' ',ErrMat[x][6])
    
                    rn = str(ErrMat_kn[x][0])+','+str(ErrMat_kn[x][1])+','+str(ErrMat_kn[x][2])+','+str(ErrMat_kn[x][3])+','+str(ErrMat_kn[x][4])+','+str(ErrMat_kn[x][5])+','+str(ErrMat_kn[x][6])+'\n'
                    file.write(rn) 
                    x += 1
                    
                except:
                    er3 += 1
                    print('knerr3 :',er3)
    
    
    file.close()
    print('len ErMat_kn ',len(ErrMat_kn),' knerr1 : ',er1,' knerr2 : ',er2,' knerr3 : ',er3,' knerr4 : ',er4)
    #==================================================== Total Error Calculation (TTE) Knn
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_TTE_KN.csv"
    file = open(path,'w')
    
    rowEr = rateR
    colEr = 3
    TTE_kn = [[0 for x in range(colEr)] for y in range(rowEr)] 
    
    r = defineClass #user Row
    c = mxknclass #user clmn userFit_knn
    erTTE = 0
    erTTE1 = 0
    
    tec_len_knn = 0
    ErrMatLen = len(ErrMat_kn)
    for i in range(0,r):
        for j in range(0,c):
            countMX = 0
            countRmse = 0
            for k in range(0,ErrMatLen ):
                if(newUserKnnMatrinx[i][j] != 0):
                    try:
                        if(newUserKnnMatrinx[i][j] == ErrMat_kn[k][0] and ErrMat_kn[k][1] == i):
                            countRmse += ErrMat[k][5]
                            countMX += ErrMat[k][6]
                    except:
                        erTTE += 1
                        print('erTTE :',erTTE)
                else:
                    break
            try:
                if(newUserKnnMatrinx[i][j] != 0 and countRmse != 0 and countMX != 0):
                    TTE_kn[tec_len_knn][0] = newUserKnnMatrinx[i][j]
                    TTE_kn[tec_len_knn][1] = countMX
                    TTE_kn[tec_len_knn][2] = countRmse
                
                    rn = str(TTE_kn[tec_len_knn][0])+','+str(TTE_kn[tec_len_knn][1])+','+str(TTE_kn[tec_len_knn][2])+'\n'
                    file.write(rn) 
                
                    tec_len_knn += 1
            except:
                erTTE1 += 1
                print('erTTE1 :',erTTE1)
    file.close()
    
    
    #==================================================== Total Error Calculation (TTE) Dt
    
    path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_TTE_DT.csv"
    file = open(path,'w')
    
    rowEr = rateR
    colEr = 3
    TTE_DT = [[0 for x in range(colEr)] for y in range(rowEr)] 
    
    r = defineClass #user Row
    c = mxDTclass #user clmn userFit_knn
    erTTE = 0
    erTTE1 = 0
    
    tec_len_dt = 0
    ErrMatLen = len(ErrMat)
    
    for i in range(0,r):
        for j in range(0,c):
            countMX = 0
            countRmse = 0
            for k in range(0,ErrMatLen ):
                if(newUserDtMatrinx[i][j] != 0):
                    try:
                        if(newUserDtMatrinx[i][j] == ErrMat[k][0] and ErrMat[k][1] == i):
                            countRmse += ErrMat[k][5]
                            countMX += ErrMat[k][6]
                    except:
                        erTTE += 1
                        print('DT_erTTE :',erTTE)
                else:
                    break
            try:
                if(newUserDtMatrinx[i][j] != 0 and countRmse != 0 and countMX != 0):
                    TTE_DT[tec_len_dt][0] = newUserDtMatrinx[i][j]
                    TTE_DT[tec_len_dt][1] = countMX
                    TTE_DT[tec_len_dt][2] = countRmse
                
                    rn = str(TTE_DT[tec_len_dt][0])+','+str(TTE_DT[tec_len_dt][1])+','+str(TTE_DT[tec_len_dt][2])+'\n'
                    file.write(rn) 
                
                    tec_len_dt += 1
            except:
                erTTE1 += 1
                print('DT_erTTE1 :',erTTE1)
    file.close()
    
    #==================================================== Execution Time
    print('Used Time',time.time() - start_time,'s')
    return ('Read All file success, Rating has been generated')