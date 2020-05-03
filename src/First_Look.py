#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Read text file and turn it into a pandas data frame
def DFfromTXT(fileName):
    txt = open(fileName,'r').read()

    # create list of lists from txt
    lines = txt.splitlines()
    xs = [l.split('\t') for l in lines]

    # join the listed proteins if there is more than one
    for x in xs:
        while (len(x) > len(xs[0])):
            x[-2:] = [', '.join(x[-2:])]

    # create a pandas DataFrame from the acquired data
    df = pd.DataFrame(xs[1:], index = range(len(xs[1:])), columns = xs[0])
    return df

# add columns containing FDR and the q-value respectively
def calcQ(df, scoreColName):
    df[scoreColName] = [float(i) for i in df[scoreColName]]
    
    # Sort by score columns
    df.sort_values(scoreColName, ascending=False, inplace = True)

    # Replace -1 by 0 and turn every element into a number
    df = df[df.Label != '-']
    df['Label'].replace(to_replace = '1', value = 1, inplace = True)
    df['Label'].replace(to_replace = '-1', value = 0, inplace = True)

    # calculate FDR
    df['FDR'] = 1 - (df['Label'].cumsum()/[i + 1 for i in range(len(df.index))])

    # calculate q-value
    df['q-val'] = df['FDR'][::-1].cummin()[::-1]
    
    return df

# add a column containing the Ranks of entries with the same id
def addRanks(df, idColName, scoreColName):
    # Sort by score to have the ranks in increasing order, and by ID to group same IDs
    df.sort_values(scoreColName, inplace = True, ascending = False)
    df.sort_values(idColName, inplace = True, kind = 'mergesort')

    # for better performance iterate over lists
    ids = list(df[idColName])
    ranks = []
    lastId = ''

    # iterate over the sorted IDs, increasing the rank for every occurence of the same ID
    for currId in ids:
        if (currId == lastId):
            currRank += 1
        else:
            currRank = 1
            lastId = currId
        ranks += [currRank]

    df['Rank'] = ranks
    return df

# Read the specified file, calculate FDR, q-value and ranks and sort it according to its score
def readAndProcess(fileName, idColName,  scoreColName):
    d = DFfromTXT(fileName)
    d1 = calcQ(d, scoreColName)
    df = addRanks(d1, idColName, scoreColName)
    df.sort_values(scoreColName, inplace = True, ascending = False)
    return df    

# Plot Pseudo ROC of the df including colums 'q-val' in x = [0,xMax] for all entries (with col 'Rank' being 1, if specified)
def pseudoROC(df, xMax, onlyFirstRank):
    if (onlyFirstRank):
        qVals = [df.loc[i, 'q-val'] for i in df.index if (df.loc[i,'q-val'] <= xMax and df.loc[i, 'Rank'] == 1)]
    else: 
        qVals = [df.loc[i, 'q-val'] for i in df.index if (df.loc[i,'q-val'] <= xMax)]
    plt.xlim(0,xMax)
    plt.ylim(0,len(qVals))
    plt.plot(qVals, range(len(qVals)))


# In[2]:


idCol = 'SpecId'
scoreCol = 'NuXL:score'

dFast = readAndProcess('../Data/1-AChernev_080219_dir_HeLa_cyt_UCGA_fast.tsv', idCol, scoreCol)
dSlow = readAndProcess('../Data/1-AChernev_080219_dir_HeLa_cyt_UCGA_slow.tsv', idCol, scoreCol)

dFast['NuXL:isXL'] = [int(x) for x in dFast['NuXL:isXL']]
dSlow['NuXL:isXL'] = [int(x) for x in dSlow['NuXL:isXL']]

dFast.head()


# In[3]:


# Plot data from slow scoring method using only rank 1
dSlXL = dSlow[dSlow['NuXL:isXL'] == 1]
dSlNoXL = dSlow[dSlow['NuXL:isXL'] == 0]
pseudoROC(dSlNoXL, 0.05, True)
pseudoROC(dSlXL, 0.05, True)


# In[4]:


# Plot data from fast scoring method using only rank 1
dFaXL = dFast[dFast['NuXL:isXL'] == 1]
dFaNoXL = dFast[dFast['NuXL:isXL'] == 0]
pseudoROC(dFaNoXL, 0.05, True)
pseudoROC(dFaXL, 0.05, True)


# In[5]:


# Plot data from fast scoring method using all ranks
pseudoROC(dFaNoXL, 0.05, False)
pseudoROC(dFaXL, 0.05, False)


# In[6]:


df = dSlow
x = [float(x) for x in df[scoreCol]]
y = [float(x) for x in df['FDR']]
plt.plot(x, y)


# In[7]:


# collect true and false training data
scores = ['NuXL:total_loss_score','NuXL:ladder_score', 'NuXL:precursor_score']
for i in df[scores]:
    df[i] = [float(x) for x in df[i]]
    
ixTrue = [x for x in df.index if (df['q-val'].iloc[x] <= 0.05 and df['Label'].iloc[x] == 1)]
ixFalse = [x for x in df.index if (df['Label'].iloc[x] == 0)]
ixFalse = rd.choices(ixFalse, k = len(ixTrue))


# In[18]:


# evaluate learning method
def evalLearning(mlObj, methodname = 'method', numTests = 500):
    ds = df[scores]
    ixRand = []
    
    for i in range(numTests):
        ixRand += [rd.randint(min(df.index), max(df.index))]
        
    prediction = mlObj.predict([list(ds.loc[ixRand[i],]) for i in range(numTests)])
    testScores = [df.loc[ixRand[i],'NuXL:score'] for i in range(numTests)]
    
    plt.xlabel('NuXL:score')
    plt.ylabel('Classification by ' + methodname)
    return plt.scatter(testScores, prediction)


# In[22]:


# train lda
from sklearn import discriminant_analysis as da
lda = da.LinearDiscriminantAnalysis()

falseTrain = df[scores].iloc[ixFalse].values.tolist()
trueTrain = df[scores].iloc[ixTrue].values.tolist()
train = falseTrain + trueTrain

classes = [0] * len(falseTrain) + [1] * len(trueTrain)
lda.fit(train, classes)

evalLearning(lda, 'linear discriminant analysis')
plt.savefig('lda_compared_to_NuXLscore.png')


# In[26]:


# train logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(train, classes)

evalLearning(lr, 'logistic regression')
plt.savefig('../results/lr_compared_to_NuXLscore.png')


# In[27]:


# train svm
from sklearn import svm
clf = svm.SVC().fit(train, classes)

evalLearning(clf, 'support vector classification')
plt.savefig('../results/svm_compared_to_NuXLscore.png')

