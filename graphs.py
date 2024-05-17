#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# In[16]:


randomTestAcc = []
randomTrainAcc = []
randomTrainSD = []
randomTestSD = []

with open('random.txt', 'r') as file:
    lines = file.readlines()
    
    i = 0 
    while i < len(lines):
        word = lines[i].strip(': \n')
        print(word)
        i+=1
        data = []
        while i < len(lines) and lines[i] != '\n':
            data.append(lines[i].strip('\n'))
            i+=1
        if word == 'accuracyTrainAverageRandom':
            randomTrainAcc = data
        elif word == 'accuracyTrainSDRandom':
            randomTrainSD = data
        elif word == 'accuracyTestAverageRandom':
            randomTestAcc = data
        elif word == 'accuracyTestSDRandom':
            randomTestSD = data
        


# In[17]:



accuracyTrainAverageRandom = []
accuracyTestAverageRandom = []
accuracyTrainSDRandom = []
accuracyTestSDRandom = []


with open('random.txt', 'r') as file:
    lines = file.readlines()

    
    currentList = None
    for line in lines:
        line = line.strip()  
        if line.endswith(':'):
            currentList = line[:-1]  # Remove the colon
        elif line:
            if currentList == 'accuracyTrainAverageRandom':
                accuracyTrainAverageRandom.append(float(line))
            elif currentList == 'accuracyTestAverageRandom':
                accuracyTestAverageRandom.append(float(line))
            elif currentList == 'accuracyTrainSDRandom':
                accuracyTrainSDRandom.append(float(line))
            elif currentList == 'accuracyTestSDRandom':
                accuracyTestSDRandom.append(float(line))

print("accuracyTrainAverageRandom:", accuracyTrainAverageRandom)
print("accuracyTestAverageRandom:", accuracyTestAverageRandom)
print("accuracyTrainSDRandom:", accuracyTrainSDRandom)
print("accuracyTestSDRandom:", accuracyTestSDRandom)


# In[18]:



accuracyTrainAverageMS = []
accuracyTestAverageMS = []
accuracyTrainSDMS = []
accuracyTestSDMS = []


with open('MS.txt', 'r') as file:
    lines = file.readlines()

    
    currentList = None
    for line in lines:
        line = line.strip()  
        if line.endswith(':'):
            currentList = line[:-1]  # Remove the colon
        elif line:
            if currentList == 'accuracyTrainAverageMS':
                accuracyTrainAverageMS.append(float(line))
            elif currentList == 'accuracyTestAverageMS':
                accuracyTestAverageMS.append(float(line))
            elif currentList == 'accuracyTrainSDMS':
                accuracyTrainSDMS.append(float(line))
            elif currentList == 'accuracyTestSDMS':
                accuracyTestSDMS.append(float(line))

print("accuracyTrainAverageMS:", accuracyTrainAverageMS)
print("accuracyTestAverageMS:", accuracyTestAverageMS)
print("accuracyTrainSDMS:", accuracyTrainSDMS)
print("accuracyTestSDMs:", accuracyTestSDMS)


# In[19]:



accuracyTrainAverageMSPos = []
accuracyTestAverageMSPos = []
accuracyTrainSDMSPos = []
accuracyTestSDMSPos = []


with open('MSP.txt', 'r') as file:
    lines = file.readlines()

    
    currentList = None
    for line in lines:
        line = line.strip()  
        if line.endswith(':'):
            currentList = line[:-1]  # Remove the colon
        elif line:
            if currentList == 'accuracyTrainAverageMSP':
                accuracyTrainAverageMSPos.append(float(line))
            elif currentList == 'accuracyTestAverageMSP':
                accuracyTestAverageMSPos.append(float(line))
            elif currentList == 'accuracyTrainSDMSP':
                accuracyTrainSDMSPos.append(float(line))
            elif currentList == 'accuracyTestSDMSP':
                accuracyTestSDMSPos.append(float(line))

print("accuracyTrainAverageMS:", accuracyTrainAverageMSPos)
print("accuracyTestAverageMS:", accuracyTestAverageMSPos)
print("accuracyTrainSDMS:", accuracyTrainSDMSPos)
print("accuracyTestSDMs:", accuracyTestSDMSPos)


# In[20]:


accuracyTrainAverageDB = []
accuracyTestAverageDB = []
accuracyTrainSDDB = []
accuracyTestSDDB = []


with open('DB.txt', 'r') as file:
    lines = file.readlines()

    
    currentList = None
    for line in lines:
        line = line.strip()  
        if line.endswith(':'):
            currentList = line[:-1]  # Remove the colon
        elif line:
            if currentList == 'accuracyTrainAverageDB':
                accuracyTrainAverageDB.append(float(line))
            elif currentList == 'accuracyTestAverageDB':
                accuracyTestAverageDB.append(float(line))
            elif currentList == 'accuracyTrainSDDB':
                accuracyTrainSDDB.append(float(line))
            elif currentList == 'accuracyTestSDDB':
                accuracyTestSDDB.append(float(line))

print("accuracyTrainAverageDB:", accuracyTrainAverageDB)
print("accuracyTestAverageDB:", accuracyTestAverageDB)
print("accuracyTrainSDDB:", accuracyTrainSDDB)
print("accuracyTestSDDB:", accuracyTestSDDB)


# In[21]:


len(accuracyTrainAverageRandom)


# In[24]:


# plot the graph

plt.figure(figsize=(12,7))

point = [i for i in range(60)]

plt.plot(point,accuracyTrainAverageRandom,linestyle='-', marker='.', color='green', alpha = 0.5)
plt.plot(point,accuracyTrainAverageMS,linestyle='-', marker='.', color='blue', alpha = 0.5)
plt.plot(point,accuracyTrainAverageMSPos,linestyle='-', marker='.', color='black', alpha = 0.5)
#plt.plot(point,accuracyTrainAverageQBC,linestyle='-', marker='.', color='red', alpha = 0.5)
#plt.plot(point,accuracyTrainAverageMI,linestyle='-', marker='.', color='yellow', alpha = 0.5)
plt.plot(point,accuracyTrainAverageDB,linestyle='-', marker='.', color='brown', alpha = 0.5)

plt.title("Plot of Training Accuracy over Number of Queries.")


p1 = mlines.Line2D([], [], color='green', marker='.', linestyle='-',
                          markersize=10, label='Random Sampling')

p2 = mlines.Line2D([], [], color='blue', marker='.', linestyle='-',
                          markersize=10, label='Margin Sampling')


p3 = mlines.Line2D([], [], color='black', marker='.', linestyle='-',
                          markersize=10, label='Margin Sampling Positive')


p6 = mlines.Line2D([], [], color='brown', marker='.', linestyle='-',
                          markersize=10, label='Density-Based')

plt.ylim(0.7,0.97)

plt.legend(handles=[p1,p2,p3,p6])



plt.xlabel("Number of Queries",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[28]:


# plot the graph

plt.figure(figsize=(12,7))

point = [i for i in range(60)]

plt.plot(point,accuracyTestAverageRandom,linestyle='-', marker='.', color='green', alpha = 0.5)
plt.plot(point,accuracyTestAverageMS,linestyle='-', marker='.', color='blue', alpha = 0.5)
plt.plot(point,accuracyTestAverageMSPos,linestyle='-', marker='.', color='black', alpha = 0.5)
#plt.plot(point,accuracyTrainAverageQBC,linestyle='-', marker='.', color='red', alpha = 0.5)
#plt.plot(point,accuracyTrainAverageMI,linestyle='-', marker='.', color='yellow', alpha = 0.5)
plt.plot(point,accuracyTestAverageDB,linestyle='-', marker='.', color='brown', alpha = 0.5)

plt.title("Plot of Testing Accuracy over Number of Queries.")


p1 = mlines.Line2D([], [], color='green', marker='.', linestyle='-',
                          markersize=10, label='Random Sampling')

p2 = mlines.Line2D([], [], color='blue', marker='.', linestyle='-',
                          markersize=10, label='Margin Sampling')


p3 = mlines.Line2D([], [], color='black', marker='.', linestyle='-',
                          markersize=10, label='Margin Sampling Positive')


p6 = mlines.Line2D([], [], color='brown', marker='.', linestyle='-',
                          markersize=10, label='Density-Based')

plt.ylim(0.8,1)

plt.legend(handles=[p1,p2,p3,p6])



plt.xlabel("Number of Queries",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[70]:




plt.figure(figsize=(12, 7))


barPositions = np.arange(6)



bars = plt.bar(barPositions, [accuracyTrainAverageRandom[14], accuracyTrainAverageMS[14],
                           accuracyTrainAverageMSPos[14], accuracyTrainAverageQBC[14],
                           accuracyTrainAverageMI[14], accuracyTrainAverageDB[14]],
        bar_width, color=['green', 'blue', 'black', 'red', 'yellow', 'brown'], alpha=0.5,
        label='Accuracy After 15 Query Points')


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}',
                ha='center', va='bottom', fontsize=12)

plt.title("Training Accuracy at 15th Query Points")
plt.xlabel("Active Learning Method", size=15)
plt.ylabel("Accuracy Value", size=15)
plt.xticks(barPositions + bar_width / 2,
           ['Random Sampling', 'Margin Sampling', 'Margin Sampling Positive',
            'Query-By-Committee', 'Mutual Information', 'Density-Based'], rotation=45)
plt.ylim(0.95, 0.97)

plt.tight_layout()
plt.show()


# In[71]:




plt.figure(figsize=(12, 7))

w = 0.40 


barPositions = np.arange(6)



bars = plt.bar(barPositions, [accuracyTrainAverageRandom[29], accuracyTrainAverageMS[29],
                           accuracyTrainAverageMSPos[29], accuracyTrainAverageQBC[29],
                           accuracyTrainAverageMI[29], accuracyTrainAverageDB[29]],
        bar_width, color=['green', 'blue', 'black', 'red', 'yellow', 'brown'], alpha=0.5,
        label='Accuracy After 30 Query Points')


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}',
                ha='center', va='bottom', fontsize=12)

plt.title("Training Accuracy at 30th Query Points")
plt.xlabel("Active Learning Method", size=15)
plt.ylabel("Accuracy Value", size=15)
plt.xticks(barPositions + bar_width / 2,
           ['Random Sampling', 'Margin Sampling', 'Margin Sampling Positive',
            'Query-By-Committee', 'Mutual Information', 'Density-Based'], rotation=45)
plt.ylim(0.95, 0.97)

plt.tight_layout()
plt.show()


# In[74]:



plt.figure(figsize=(12, 7))

w = 0.40 


barPositions = np.arange(6)



bars = plt.bar(barPositions, [accuracyTrainAverageRandom[19], accuracyTrainAverageMS[19],
                           accuracyTrainAverageMSPos[19], accuracyTrainAverageQBC[19],
                           accuracyTrainAverageMI[19], accuracyTrainAverageDB[19]],
        bar_width, color=['green', 'blue', 'black', 'red', 'yellow', 'brown'], alpha=0.5,
        label='Accuracy After 20 Query Points')


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}',
                ha='center', va='bottom', fontsize=12)

plt.title("Training Accuracy at 20th Query Points")
plt.xlabel("Active Learning Method", size=15)
plt.ylabel("Accuracy Value", size=15)
plt.xticks(barPositions + bar_width / 2,
           ['Random Sampling', 'Margin Sampling', 'Margin Sampling Positive',
            'Query-By-Committee', 'Mutual Information', 'Density-Based'], rotation=45)
plt.ylim(0.95, 0.97)

plt.tight_layout()
plt.show()


# In[83]:


plt.figure(figsize=(12, 7))

w = 0.40 


barPositions = np.arange(6)



bars = plt.bar(barPositions, [270.41, 281.00,
                           283.77,291.18,
                          2523.47, 298.93],
        bar_width, color=['green', 'blue', 'black', 'red', 'yellow', 'brown'], alpha=0.5,
        label='Time taken')




plt.title("Time taken for 30 Queries")
plt.xlabel("Active Learning Method", size=15)
plt.ylabel("Time (seconds)", size=15)
plt.xticks(barPositions + bar_width / 2,
           ['Random Sampling', 'Margin Sampling', 'Margin Sampling Positive',
            'Query-By-Committee', 'Mutual Information', 'Density-Based'], rotation=45)
#plt.ylim(0.95, 0.97)


plt.show()


# In[ ]:




