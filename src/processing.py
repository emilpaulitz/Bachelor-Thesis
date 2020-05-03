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
    #df[scoreColName] = [float(i) for i in df[scoreColName]]
    
    # Sort by score columns
    df.sort_values(scoreColName, ascending=False, inplace = True)

    # Replace -1 by 0 and turn every element into a number
    df['Label'].replace(to_replace = -1, value = 0, inplace = True)

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
    d1 = strToNum(d)
    d2 = calcQ(d1, scoreColName)
    df = addRanks(d2, idColName, scoreColName)
    df.sort_values(scoreColName, inplace = True, ascending = False)
    return df   

# convert every string col into an int or float if possible
def strToNum(df):
    for col in df:
        try:
            df[col] = [int(i) for i in df[col]]
        except ValueError:
            try:
                df[col] = [float(i) for i in df[col]]
            except ValueError:
                continue
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
