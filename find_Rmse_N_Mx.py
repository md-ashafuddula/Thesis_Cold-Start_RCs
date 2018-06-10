# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 23:55:09 2017

@author: Nezamul Islam A R
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import csv
import time
import numpy as np 
import math

path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ErrMat_dt.csv"
file = open(path,newline='')

read = csv.reader(file)
testSet = [] #empty list

for row in read:
    #print(row)
    feature_5 = float(row[3]) # 5(av-rate)**2 >> RMSE 3(avg)
    feature_6 = float(row[4]) # (rate-1)**2 >>MAX 4(rating)
    testSet.append([feature_5,feature_6]) #append to column into data array
    
file.close()

maeCount = 0
maxMaeCount = 0

count_av_rate = 0
count_rate = 0
for i in range(len(testSet)):
    if(testSet[i][1] >= 3):
        count_av_rate += (testSet[i][0] - testSet[i][1])**2
        count_rate += (testSet[i][0])**2
                      
        maeCount += abs(testSet[i][0] - testSet[i][1])
        maxMaeCount += abs(testSet[i][0])

#print(count_av_rate," ",len(testSet))
rmseDT = math.sqrt(count_av_rate / len(testSet))
maeDT = maeCount/len(testSet)

print('RMSE DT : ',rmseDT)
print('MAE DT : ',maeDT)
#print(count_rate," ",len(testSet))
maxEr = math.sqrt(count_rate / len(testSet))
maxMae = maxMaeCount / len(testSet)
print('Max RMSE DT : ',maxEr)
print('Max MAE DT : ',maxMae)
print('testSet  - length = ',len(testSet))
#============================================
path = "H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/2_ErrMat_kn.csv"
file = open(path,newline='')

read = csv.reader(file)
testSet1 = [] #empty list

for row in read:
    #print(row)
    feature_5 = float(row[3]) # 5(av-rate)**2 >> RMSE 3(avg)
    feature_6 = float(row[4]) # (rate-1)**2 >>MAX 4(rating)
    testSet1.append([feature_5,feature_6]) #append to column into data array
    
file.close()

count_av_rate = 0
count_rate = 0

maeCount = 0
maxMaeCount = 0

for i in range(len(testSet1)):
    if(testSet1[i][1] >= 3):
        count_av_rate += (testSet1[i][0] - testSet1[i][1])**2
        count_rate += (testSet1[i][0])**2
                      
        maeCount += abs(testSet1[i][0] - testSet1[i][1])
        maxMaeCount += abs(testSet1[i][0])

#print(count_av_rate," ",len(testSet))
rmseKn = math.sqrt(count_av_rate / len(testSet1))

#print(count_av_rate," ",len(testSet))
rmseKn = math.sqrt(count_av_rate / len(testSet1))
maeKn = maeCount/len(testSet1)

##for i in range(len(testSet1)):
print('RMSE KNN : ',rmseKn)
print('MAE KNN : ',maeKn)
#
#print(count_rate," ",len(testSet))
maxErKn = math.sqrt(count_rate / (len(testSet1) ))
maxMAEKn = maxMaeCount / (len(testSet1))
    
print('Max Err KNN : ',maxErKn)
print('Max Err KNN : ',maxMAEKn)

print('testSet 1 - length = ',len(testSet1))

#=============plot   
#plt.scatter(2,rmse,marker = 'x',s = 60,linewidths = 100)
#plt.scatter(4,maxEr,marker = 'o',s =2,linewidths = 10)
#
#plt.scatter(8,rmseKn,marker = 'x',s = 60,linewidths = 100)
#plt.scatter(10,maxErKn,marker = 'o',s =2,linewidths = 10)
#
#plt.xlim([0, 12])
#plt.ylim([0, 7])
#fig = plt.figure(figsize=(5,6))
#ax1 = fig.add_subplot(111)
#
#ax1.bar([1,2,3,4],[rmse,maxEr,rmseKn,maxErKn])
#plt.show()
 