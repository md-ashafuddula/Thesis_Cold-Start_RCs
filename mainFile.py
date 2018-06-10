# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:04:21 2017

@author: Nezamul Islam A R
"""
import ElbowFinder
import readAllFiles
import clusterGeneratesFile

print('Main File ...')
flagMain = int(input('Enter 1 to choose manual class no.\nEnter 2 to use Elbow method : '))

if(flagMain == 1):
    classNo = int(input('Enter class No : '))
elif(flagMain == 2):
    init = 10#int(input('Enter Elbow Init position : ')) #0
    final = 30#int(input('Enter Elbow Final position : ')) #708
    getRes = ElbowFinder.elbowFind(init,final)
    print(getRes)
    classNo = int(input('Choose Class No. : '))
else:
    print('Alert!! Wrong Input .')
#generates Cluster
print(clusterGeneratesFile.clusterGen(classNo))
#generates Rating
print(readAllFiles.readFiles(classNo))
#ENd