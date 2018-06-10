tic;
mainFileName = 'H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/garbageFile/userDemoGraphUpdate.csv';
mainUserFile = csvread(mainFileName) %fetching User DataSet
testFold = mainUserFile(1:235,1:4)
trainFold = mainUserFile(236:943,1:4)

testFile = zeros(235,3)
trainFile = zeros(708,3)

featureMat = testFold(:,2:4) % Only Features
[u,s,v] = svd(featureMat,'econ') %s
v = v(:,1:2)
Xmat = featureMat*v
testFile(:,1:1) = testFold(:,1:1)
testFile(:,2:3) = Xmat(:,1:2)
csvwrite('H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_testdataSet.csv',testFile)

featureMat = trainFold(:,2:4) % Only Features
[u,s,v] = svd(featureMat,'econ') %s
v = v(:,1:2)
Xmat = featureMat*v
trainFile(:,1:1) = trainFold(:,1:1)
trainFile(:,2:3) = Xmat(:,1:2)
csvwrite('H:/Mine 8th sem/8 sem Thesis/New Work/Implementation/Final KNN Implementation/2/Data_R/3/1_trainSet.csv',trainFile)

toc;

%time = 8.33 s