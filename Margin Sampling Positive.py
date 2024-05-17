#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from random import sample
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import svm
import time


# In[ ]:





# ## Read in the train feature and train label file

# In[30]:


data = pd.ExcelFile('M. musculus_Constructed_RFInput.xlsx')
data


# In[31]:


# trainX

dataTestFeature = data.parse(data.sheet_names[3])
dataTestFeature


# In[32]:


# trainY

dataTestLabel = data.parse(data.sheet_names[0])
dataTestLabel


# In[33]:


# testX

dataTrainLabel = data.parse(data.sheet_names[1])
dataTrainLabel


# In[34]:


# testY

dataTrainFeature = data.parse(data.sheet_names[2])
dataTrainFeature


# In[ ]:





# ## PCA plot to visualise distribution of samples between two groups

# In[35]:



trainX = dataTrainFeature
trainY = dataTrainLabel
testX = dataTestFeature
testY = dataTestLabel

TrainSet = pd.concat([trainX, trainY],axis=1)
TestSet = pd.concat([testX, testY], axis = 1)

Train = TrainSet.sample(n=50, random_state=20)
trainChosen = Train.index
remainingTrain = TrainSet.drop(index=trainChosen)

Test = pd.concat([TestSet, remainingTrain],axis=0)



# In[ ]:





# In[36]:


#int(trainY.values[0])


# In[37]:



matrixTrain = trainX
matrixTrain = StandardScaler().fit_transform(matrixTrain)
trainDF = pd.DataFrame(matrixTrain)
df2C = PCA(n_components = 2)
trainDF2 = df2C.fit_transform(matrixTrain)
trainDF2Plot = pd.DataFrame(data = trainDF2, columns = ['PCA1', 'PCA2'])


# In[38]:


trainDF2Plot


# In[40]:


# PCA Plot for Train Data
plt.figure()
plt.figure(figsize = (12,12))
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("PCA1", fontsize=20)
plt.ylabel("PCA2",fontsize=20)


labels = [1,0]
colors = ['g','b']
for label,color in zip(labels, colors):
    for index in range(len(trainY.values)):
    #print(index)
        if label == int((trainY.values)[index]):
      #print((trainY.values)[index])
          plt.scatter(trainDF2Plot["PCA1"][index], trainDF2Plot["PCA2"][index],c= color ,s = 50)

p1 = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=10, label='1 - Interact')

p2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='0 - Does not Interact')

plt.title('PCA Plot for Training Data', fontsize=25)
plt.legend(handles = [p1,p2], fontsize=15)
plt.show()


# In[41]:



matrixTest = testX
matrixTest = StandardScaler().fit_transform(matrixTest)
testDF = pd.DataFrame(matrixTest)
df2C = PCA(n_components = 2)
testDF2 = df2C.fit_transform(matrixTest)
testDF2Plot = pd.DataFrame(data = testDF2, columns = ['PCA1', 'PCA2'])


# In[42]:


# PCA Plot for Test Data
plt.figure()
plt.figure(figsize = (12,12))

plt.xlabel("PCA1", fontsize =20)
plt.ylabel("PCA2", fontsize =20)


labels = [1,0]
colors = ['y','b']
for label,color in zip(labels, colors):
    for index in range(len(testY)):
        if label == int((testY)[index]):
            plt.scatter(testDF2Plot["PCA1"][index], testDF2Plot["PCA2"][index],c= color ,s = 50)

p1 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='1 - Interact')

p2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='0 - Does not interact')


plt.title('PCA Plot for Testing Data', fontsize=25)
plt.legend(handles = [p1,p2], fontsize=15)
plt.show()


# In[ ]:





# ## Active Learning - Margin Sampling

# In[43]:


def MSPosPick(train, test):
    indexList = test.index
    trainX = train[train.columns[:-1]].to_numpy()
    trainY = train[train.columns[-1]].to_numpy()

    testX = test[test.columns[:-1]].to_numpy()
    testY = test[test.columns[-1]].to_numpy()

    clf = svm.SVC(kernel='rbf')
    clf.fit(trainX, trainY)

  # distances of unlabeled data points from hyperplane
    dists = clf.decision_function(testX)

  # get all data points of positive class (1 for positive values, 0 for negative values)
    pos_ind = np.where(dists > 0)[0]

  # pick closest unlabeled data point among just the positive class data points
    pos_dists = dists[pos_ind]
    min_pos_ind = np.argmin(pos_dists)
    ms_pos_ind = pos_ind[min_pos_ind]

    return indexList[ms_pos_ind]


# In[51]:


start = time.time()
accuracyTrainAverageMSPos = []
accuracyTestAverageMSPos = []
accuracyTrainSDMSPos = []
accuracyTestSDMSPos = []

train = Train
test = Test
chosenMSPos = []

for random in range(60):


    # implement Margin sampling

    pickMSRow = MSPosPick(train, test)
    print(pickMSRow)
    
    train = pd.concat([train, test.loc[[pickMSRow]]], axis=0)
    test.drop(pickMSRow, inplace=True)
    chosenMSPos.append(pickMSRow)


    lossTrainSubset = []
    lossTestSubset = []

    for i in range(5):  # 5 times for different random seeds
        # to have 5 folds to do cross validation

        kFold = KFold(n_splits = 5, shuffle = True, random_state = i+10)


        # run across 5 iterations / 5 folds
        predictions = []
        actual = []


        trainX = train[train.columns[:-1]].to_numpy()
        trainY = train[train.columns[-1]].to_numpy()

        testX = test[test.columns[:-1]].to_numpy()
        testY = test[test.columns[-1]].to_numpy()

        for trainIndex, valIndex in kFold.split(trainX):
            xTrain, xVal = trainX[trainIndex], trainX[valIndex]
            yTrain, yVal = trainY[trainIndex], trainY[valIndex]
            rf = RandomForestClassifier()
            rf.fit(xTrain, yTrain)
            yPred = rf.predict(xVal)
            predictions += yPred.tolist()
            actual += yVal.tolist()

        loss = accuracy_score(actual,predictions)
        lossTrainSubset.append(loss)

        rf.fit(trainX, trainY)
        predictTest = rf.predict(testX)
        lossTest = accuracy_score(predictTest, testY.tolist())

        lossTestSubset.append(lossTest)

    accuracyTrainAverageMSPos.append(np.mean(lossTrainSubset))
    accuracyTrainSDMSPos.append(np.std(lossTrainSubset))
    print(np.mean(lossTrainSubset))
    print(np.std(lossTrainSubset))
    accuracyTestAverageMSPos.append(np.mean(lossTestSubset))
    accuracyTestSDMSPos.append(np.std(lossTestSubset))
end = time.time()
elapsed = end-start
print(elapsed)


# In[52]:



trainX = dataTrainFeature
trainY = dataTrainLabel
testX = dataTestFeature
testY = dataTestLabel

Train = pd.concat([trainX, trainY],axis=1)
Test = pd.concat([testX, testY], axis = 1)


# In[68]:



# color the points chosen by margin sampling


plt.figure()
plt.figure(figsize = (12,12))

plt.xlabel("PCA1", fontsize =20)
plt.ylabel("PCA2", fontsize =20)


labels = [1,0]
colors = ['y','b']
for label,color in zip(labels, colors):
    for index in range(len(testY.values)):
    #print(index)
        if label == int((testY.values)[index]) and index in chosenMSPos:
            plt.scatter(testDF2Plot["PCA1"][index], testDF2Plot["PCA2"][index],c= 'purple', edgecolor = 'none' ,s = 50)
        elif label == int((testY.values)[index]) and index not in chosenMSPos:
            plt.scatter(testDF2Plot["PCA1"][index], testDF2Plot["PCA2"][index],c = 'none', edgecolor=color,s = 50)

p1 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='1 - Interact')

p2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='0 - Does not interact')

p3 = mlines.Line2D([], [], color='purple', marker='o', linestyle='None',
                          markersize=10, label='Margin Sampling Positive Selection')


plt.title('PCA Plot for Testing Data', fontsize=25)
plt.legend(handles = [p1,p2,p3], fontsize=15)
plt.show()


# In[54]:


# plot the graph

plt.figure(figsize=(12,7))

point = [i for i in range(60)]

plt.errorbar(point,accuracyTrainAverageMSPos, accuracyTrainSDMSPos,linestyle='-', marker='.', color='green', alpha = 0.5)
plt.errorbar(point,accuracyTestAverageMSPos, accuracyTestSDMSPos,linestyle='-', marker='.', color='blue', alpha = 0.5)

plt.title("Plot of Accuracy over data points.")


p1 = mlines.Line2D([], [], color='green', marker='.', linestyle='-',
                          markersize=10, label='Train - Margin Sampling')

p2 = mlines.Line2D([], [], color='blue', marker='.', linestyle='-',
                          markersize=10, label='Test - Margin Sampling')

plt.ylim(0.7,1.01)

plt.legend(handles=[p1,p2])



plt.xlabel("Number of Queries",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[56]:



info = ['accuracyTrainAverageMSP','accuracyTestAverageMSP','accuracyTrainSDMSP','accuracyTestSDMSP']
with open('MSP.txt','w') as file:
    for name, data in zip(info, [accuracyTrainAverageMSPos,accuracyTestAverageMSPos,accuracyTrainSDMSPos,accuracyTestSDMSPos]):
        file.write(f"{name}:\n")
        for item in data:
            file.write(f"{item}\n")
        file.write('\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




