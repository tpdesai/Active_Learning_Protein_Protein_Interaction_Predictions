#!/usr/bin/env python
# coding: utf-8

# In[22]:


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
import time


# In[ ]:





# ## Read in the train feature and train label file

# In[23]:


data = pd.ExcelFile('M. musculus_Constructed_RFInput.xlsx')
data


# In[24]:


# trainX

dataTestFeature = data.parse(data.sheet_names[3])
dataTestFeature


# In[25]:


# trainY

dataTestLabel = data.parse(data.sheet_names[0])
dataTestLabel


# In[26]:


# testX

dataTrainLabel = data.parse(data.sheet_names[1])
dataTrainLabel


# In[27]:


# testY

dataTrainFeature = data.parse(data.sheet_names[2])
dataTrainFeature


# In[ ]:





# ## PCA plot to visualise distribution of samples between two groups

# In[28]:



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





# In[29]:


int(trainY.values[110])


# In[30]:



matrixTrain = trainX
matrixTrain = StandardScaler().fit_transform(matrixTrain)
trainDF = pd.DataFrame(matrixTrain)
df2C = PCA(n_components = 2)
trainDF2 = df2C.fit_transform(matrixTrain)
trainDF2Plot = pd.DataFrame(data = trainDF2, columns = ['PCA1', 'PCA2'])


# In[31]:


trainDF2Plot


# In[32]:


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


# In[33]:



matrixTest = testX
matrixTest = StandardScaler().fit_transform(matrixTest)
testDF = pd.DataFrame(matrixTest)
df2C = PCA(n_components = 2)
testDF2 = df2C.fit_transform(matrixTest)
testDF2Plot = pd.DataFrame(data = testDF2, columns = ['PCA1', 'PCA2'])


# In[34]:


# PCA Plot for Test Data
plt.figure()
plt.figure(figsize = (12,12))

plt.xlabel("PCA1", fontsize =20)
plt.ylabel("PCA2", fontsize =20)


labels = [1,0]
colors = ['y','b']
for label,color in zip(labels, colors):
    for index in range(len(testY.values)):
        if label == int((testY.values)[index]):
            plt.scatter(testDF2Plot["PCA1"][index], testDF2Plot["PCA2"][index],c= color ,s = 50)

p1 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='1 - Interact')

p2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='0 - Does not interact')


plt.title('PCA Plot for Testing Data', fontsize=25)
plt.legend(handles = [p1,p2], fontsize=15)
plt.show()


# In[ ]:





# ## Active Learning - Density Based Sampling

# In[35]:


def DBSPick(train, test):

    indexList = test.index
    print(indexList)
    
    trainX = train[train.columns[:-1]].to_numpy()
    trainY = train[train.columns[-1]].to_numpy()

    testX = test[test.columns[:-1]].to_numpy()
    testY = test[test.columns[-1]].to_numpy()

  # initialize array for calculated values (based on density-based sampling formula from lecture) of all points in current test set (unlabeled pool)
    dbs_values = []

  # iterate through each point in current test set
    len_U = len(testX)
    beta = 1

  # calculating utility (based on uncertainty sampling and calculating confidence)
    rf = RandomForestClassifier()
    rf.fit(trainX, trainY)

    all_probs = rf.predict_proba(testX)

    confs = []
    for x_prob in all_probs:
        confs.append(1-np.max(x_prob))

    for i in range(len(testX)):
        x_u = testX[i]
        confidence = confs[i]

        sim_score_sum = 0

    # iterate through all the other points in the test set
        for x_p in testX:
            if (x_u == x_p).all() == False:
                euclidean_dist = np.linalg.norm(x_u-x_p)
                sim_score_sum += euclidean_dist

        total_value = (((1/len_U) * sim_score_sum)**beta) * confidence
        dbs_values.append(total_value)

  # get index of point that generates the maximum value
    dbs_ind = np.argmax(dbs_values)
    print(dbs_ind)
    return indexList[dbs_ind]


# In[36]:



trainX = dataTrainFeature
trainY = dataTrainLabel
testX = dataTestFeature
testY = dataTestLabel

Train = pd.concat([trainX, trainY],axis=1)
Test = pd.concat([testX, testY], axis = 1)


# In[42]:


start = time.time()
accuracyTrainAverageDS = []
accuracyTestAverageDS = []
accuracyTrainSDDS = []
accuracyTestSDDS = []

train = Train
test = Test
chosenDS = []

for random in range(60):


    # implement Density Based Sampling

    pickMSRow = DBSPick(train, test)
    if pickMSRow in chosenDS:
        continue
    print(pickMSRow)
    
    train = pd.concat([train, test.loc[[pickMSRow]]], axis=0)
    test.drop(pickMSRow, inplace=True)
    chosenDS.append(pickMSRow)


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

    accuracyTrainAverageDS.append(np.mean(lossTrainSubset))
    accuracyTrainSDDS.append(np.std(lossTrainSubset))
    print(np.mean(lossTrainSubset))
    print(np.std(lossTrainSubset))
    accuracyTestAverageDS.append(np.mean(lossTestSubset))
    accuracyTestSDDS.append(np.std(lossTestSubset))
end = time.time()
elapsed = end-start
print(elapsed)


# In[43]:



trainX = dataTrainFeature
trainY = dataTrainLabel
testX = dataTestFeature
testY = dataTestLabel

Train = pd.concat([trainX, trainY],axis=1)
Test = pd.concat([testX, testY], axis = 1)


# In[44]:



# color the points chosen by  density based

plt.figure()
plt.figure(figsize = (12,12))

plt.xlabel("PCA1", fontsize =20)
plt.ylabel("PCA2", fontsize =20)


labels = [1,0]
colors = ['y','b']
for label,color in zip(labels, colors):
    for index in range(len(testY.values)):
    #print(index)
        if label == int((testY.values)[index]) and index in chosenDS:
            plt.scatter(testDF2Plot["PCA1"][index], testDF2Plot["PCA2"][index],c= 'purple' ,s = 50)
        elif label == int((testY.values)[index]) and index not in chosenDS:
            plt.scatter(testDF2Plot["PCA1"][index], testDF2Plot["PCA2"][index],c= color ,s = 50)

p1 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='1 - Interact')

p2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='0 - Does not interact')

p3 = mlines.Line2D([], [], color='purple', marker='o', linestyle='None',
                          markersize=10, label='Density-Based Selection')


plt.title('PCA Plot for Testing Data', fontsize=25)
plt.legend(handles = [p1,p2,p3], fontsize=15)
plt.show()


# In[45]:


# plot the graph

plt.figure(figsize=(12,7))

point = [i for i in range(60)]

plt.errorbar(point,accuracyTrainAverageDS, accuracyTrainSDDS,linestyle='-', marker='.', color='green', alpha = 0.5)
plt.errorbar(point,accuracyTestAverageDS, accuracyTestSDDS,linestyle='-', marker='.', color='blue', alpha = 0.5)

plt.title("Plot of Accuracy over data points.")


p1 = mlines.Line2D([], [], color='green', marker='.', linestyle='-',
                          markersize=10, label='Train - Density-Based Sampling')

p2 = mlines.Line2D([], [], color='blue', marker='.', linestyle='-',
                          markersize=10, label='Test - Density-Based Sampling')

plt.ylim(0.7,1.01)

plt.legend(handles=[p1,p2])



plt.xlabel("Number of Queries",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[47]:



info = ['accuracyTrainAverageDB','accuracyTestAverageDB','accuracyTrainSDDB','accuracyTestSDDB']
with open('DB.txt','w') as file:
    for name, data in zip(info, [accuracyTrainAverageDS,accuracyTestAverageDS,accuracyTrainSDDS,accuracyTestSDDS]):
        file.write(f"{name}:\n")
        for item in data:
            file.write(f"{item}\n")
        file.write('\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




