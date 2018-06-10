# Thesis_Cold-Start_RCs
Thesis Field(Col-Start Recommendation System,Machine Learning)My Thesis file .That I tested and worked with

Steps:
1)process Dataset (MovieLens DS, from grouplens org)
2)4 fold Dataset, where 1/4 are used as test and 3/4 are used as a tain data
3)apply svd in train and test dataset in matlab.
I used matlab 2017 version.
4)run main file which calls other steps
  a) find elbow
  b) take the no. of cluster make cluster with train data set
  c) Classification (KNN & DT) are used to classify new user or test dataset
  d) rating prediction for test data or new users, from which are seen by same class and rated by more or equal to 3 and at least seen by 5 users.
