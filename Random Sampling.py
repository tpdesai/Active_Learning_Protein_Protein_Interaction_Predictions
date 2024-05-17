#!/usr/bin/env python
# coding: utf-8

# In[85]:


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

# In[86]:


data = pd.ExcelFile('M. musculus_Constructed_RFInput.xlsx')
data


# In[87]:


# trainX

dataTestFeature = data.parse(data.sheet_names[3])
dataTestFeature


# In[88]:


# trainY

dataTestLabel = data.parse(data.sheet_names[0])
dataTestLabel


# In[89]:


# testX

dataTrainLabel = data.parse(data.sheet_names[1])
dataTrainLabel


# In[90]:


# testY

dataTrainFeature = data.parse(data.sheet_names[2])
dataTrainFeature


# In[ ]:





# ## PCA plot to visualise distribution of samples between two groups

# In[ ]:





# In[91]:



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



# In[92]:


remainingTrain


# In[93]:


remainingTrain


# In[ ]:





# In[94]:


int(trainY.values[110])


# In[95]:


Train


# In[96]:


Test


# In[97]:



matrixTrain = trainX
matrixTrain = StandardScaler().fit_transform(matrixTrain)
trainDF = pd.DataFrame(matrixTrain)
df2C = PCA(n_components = 2)
trainDF2 = df2C.fit_transform(matrixTrain)
trainDF2Plot = pd.DataFrame(data = trainDF2, columns = ['PCA1', 'PCA2'])


# In[98]:


trainDF2Plot


# In[99]:


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


# In[100]:



matrixTest = testX
matrixTest = StandardScaler().fit_transform(matrixTest)
testDF = pd.DataFrame(matrixTest)
df2C = PCA(n_components = 2)
testDF2 = df2C.fit_transform(matrixTest)
testDF2Plot = pd.DataFrame(data = testDF2, columns = ['PCA1', 'PCA2'])


# In[101]:


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





# ## Passive Learning (Random Sampling)

# In[102]:


# preparing test and train sets for random sampling

trainXRandom = dataTrainFeature
trainYRandom = dataTrainLabel
testXRandom = dataTestFeature
testYRandom = dataTestLabel

Train = pd.concat([trainXRandom, trainYRandom],axis=1)
Test = pd.concat([testXRandom, testYRandom], axis = 1)


# In[ ]:





# In[114]:


start = time.time()
accuracyTrainAverageRandom = []
accuracyTestAverageRandom = []
accuracyTrainSDRandom = []
accuracyTestSDRandom = []

train = Train
test = Test
chosen = []

for random in range(60):

    pickRandomRow = test.sample(n=1)
    test.drop(pickRandomRow.index,inplace=True)
    train = pd.concat([train, pickRandomRow], axis=0)
    chosen.append(pickRandomRow.index)


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

    accuracyTrainAverageRandom.append(np.mean(lossTrainSubset))
    accuracyTrainSDRandom.append(np.std(lossTrainSubset))
    print(np.mean(lossTrainSubset))
    print(np.std(lossTrainSubset))
    accuracyTestAverageRandom.append(np.mean(lossTestSubset))
    accuracyTestSDRandom.append(np.std(lossTestSubset))
end = time.time()

elapsed = end - start
print(elapsed)


# In[115]:


# plot the graph

plt.figure(figsize=(12,7))

point = [i for i in range(60)]

plt.errorbar(point,accuracyTrainAverageRandom, accuracyTrainSDRandom,linestyle='-', marker='.', color='green', alpha = 0.5)
plt.errorbar(point,accuracyTestAverageRandom, accuracyTestSDRandom,linestyle='-', marker='.', color='blue', alpha = 0.5)

plt.title("Plot of Accuracy over data points.")


p1 = mlines.Line2D([], [], color='green', marker='.', linestyle='-',
                          markersize=10, label='Train - Random Sampling')

p2 = mlines.Line2D([], [], color='blue', marker='.', linestyle='-',
                          markersize=10, label='Test - Random Sampling')

plt.ylim(0.8,1.01)

plt.legend(handles=[p1,p2])



plt.xlabel("Data Points",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[116]:


chosen[0]


# In[106]:


# plot the graph for training accuracy STD

plt.figure(figsize=(12,7))

point = [i for i in range(30)]

plt.plot(point,accuracyTrainSDRandom, linestyle='-', marker='.', color='purple', alpha = 0.5)

plt.title("Plot of Standard Deviation over number of Queries.")


p1 = mlines.Line2D([], [], color='purple', marker='.', linestyle='-',
                          markersize=10, label='Train - Random Sampling')

#plt.ylim(0.95,1.01)

plt.legend(handles=[p1])



plt.xlabel("Number of Queries",size =15)
plt.ylabel("STD Value",size=15)


# In[107]:


# plot the graph for training accuracy

plt.figure(figsize=(12,7))

point = [i for i in range(30)]

plt.plot(point,accuracyTrainAverageRandom, linestyle='-', marker='.', color='purple', alpha = 0.5)

plt.title("Plot of Accuracy over Number of Queries.")


p1 = mlines.Line2D([], [], color='purple', marker='.', linestyle='-',
                          markersize=10, label='Train - Random Sampling')

plt.ylim(0.8,1.01)

plt.legend(handles=[p1])



plt.xlabel("Number of Queries",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[108]:


# plot the graph for testing accuracy

plt.figure(figsize=(12,7))

point = [i for i in range(30)]

plt.plot(point,accuracyTestAverageRandom, linestyle='-', marker='.', color='purple', alpha = 0.5)

plt.title("Plot of Accuracy over Number of Queries.")


p1 = mlines.Line2D([], [], color='purple', marker='.', linestyle='-',
                          markersize=10, label='Test - Random Sampling')

plt.ylim(0.8,1.01)

plt.legend(handles=[p1])



plt.xlabel("Number of Queries",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[109]:


# plot the graph for testing accuracy STD

plt.figure(figsize=(12,7))

point = [i for i in range(30)]

plt.plot(point,accuracyTestSDRandom, linestyle='-', marker='.', color='purple', alpha = 0.5)

plt.title("Plot of Standard Deviation over Number of Queries.")


p1 = mlines.Line2D([], [], color='purple', marker='.', linestyle='-',
                          markersize=10, label='Test - Random Sampling')

#plt.ylim(0.5,1.01)

plt.legend(handles=[p1])



plt.xlabel("Number of Queries",size =15)
plt.ylabel("STD Value",size=15)


# In[110]:



trainX = dataTrainFeature
trainY = dataTrainLabel
testX = dataTestFeature
testY = dataTestLabel

Train = pd.concat([trainX, trainY],axis=1)
Test = pd.concat([testX, testY], axis = 1)


# In[111]:


'''
Train = pd.concat([trainXRandom, trainYRandom],axis=1)
Test = pd.concat([testXRandom, testYRandom], axis = 1)
'''


# In[121]:


# PCA Plot for Test Data
plt.figure()
plt.figure(figsize = (12,12))

plt.xlabel("PCA1", fontsize =20)
plt.ylabel("PCA2", fontsize =20)


labels = [1,0]
colors = ['y','b']
for label,color in zip(labels, colors):
    for index in range(len(testY)):
        if label == int((testY)[index]) and index not in chosen:
            plt.scatter(testDF2Plot["PCA1"][index], testDF2Plot["PCA2"][index],c= color ,s = 50)
        elif label == int((testY)[index]) and index in chosen:
            plt.scatter(testDF2Plot["PCA1"][index], testDF2Plot["PCA2"][index],c= 'purple' ,s = 50)

p1 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='1 - Interact')

p2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='0 - Does not interact')

p3 = mlines.Line2D([], [], color='purple', marker='o', linestyle='None',
                          markersize=10, label='Random Sampling Selection')


plt.title('PCA Plot for Testing Data', fontsize=25)
plt.legend(handles = [p1,p2,p3], fontsize=15)
plt.show()


# In[118]:



info = ['accuracyTrainAverageRandom','accuracyTestAverageRandom','accuracyTrainSDRandom','accuracyTestSDRandom']
with open('random.txt','w') as file:
    for name, data in zip(info, [accuracyTrainAverageRandom,accuracyTestAverageRandom,accuracyTrainSDRandom,accuracyTestSDRandom]):
        file.write(f"{name}:\n")
        for item in data:
            file.write(f"{item}\n")
        file.write('\n')


# In[117]:


print(len(accuracyTrainAverageRandom))


# In[ ]:





# In[ ]:




