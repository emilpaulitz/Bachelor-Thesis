import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn import svm
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from cycler import cycler

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
def calcQ(df, scoreColName, labelColName = 'Label'):
    
    # Sort by score column
    df.sort_values(scoreColName, ascending=False, inplace = True)

    # Replace -1 by 0
    df[labelColName].replace(to_replace = -1, value = 0, inplace = True)

    # calculate FDR
    df['FDR'] = 1 - (df[labelColName].cumsum()/[i + 1 for i in range(len(df.index))])

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
    print('file read')
    d1 = strToNum(d)
    print('strings converted to numbers')
    d2 = calcQ(d1, scoreColName)
    print('q-values estimated')
    df = addRanks(d2, idColName, scoreColName)
    print('ranks computed')
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

# Plot Pseudo ROC of the df including entries with q in x = [0,xMax] and return area under the curve
def pseudoROC(df, xMax = 0.05, onlyFirstRank = True, qColName = 'q-val', rankColName = 'Rank', title = '', label = ''):
    if (onlyFirstRank):
        qVals = [df.loc[i, qColName] for i in df.index if (df.loc[i,qColName] <= xMax and df.loc[i, rankColName] == 1)]
    else: 
        qVals = [df.loc[i, qColName] for i in df.index if (df.loc[i,qColName] <= xMax)]
    plt.xlim(0,xMax)
    plt.ylim(0,len(qVals))
    plt.title(title)
    plt.plot(qVals, range(len(qVals)), label = label)
    return auc(qVals, range(len(qVals)))

# plot pseudo ROCs of maximum, median and minimum of given q-values
def pseudoROCmulti(lss, xMax = 0.05, title = '', labels = ['maximum', 'minimum', 'median']):
    l = max([len(ls) for ls in lss])
    for ls in lss:
        ls += [float('NaN')] * (l - len(ls))
    mx = [min(elem) for elem in zip(*lss)]
    mn = [max(elem) for elem in zip(*lss)]
    md = [np.median(elem) for elem in zip(*lss)]
    plt.xlim(0, xMax)
    plt.ylim(0, len([x for x in mx if(x <= xMax)]))
    plt.plot(mx, range(len(mx)), label = labels[0])
    plt.plot(mn, range(len(mn)), label = labels[1])
    plt.plot(md, range(len(md)), label = labels[2])
    plt.legend(loc = 'best')
    
# plot pseudoROCs for XLs and Non XLs and return the areas under the curves
def evalXL(df, printResult = True):
    
    nXL = pseudoROC(df[df['NuXL:isXL'] == 0], label = 'not cross-linked')
    XL = pseudoROC(df[df['NuXL:isXL'] == 1], label = 'cross-linked')
    plt.legend(loc = 'best')
    
    if (printResult):
        print('AUC for Non-cross-linked PSMs: ' + str(nXL))
        print('AUC for Cross-linked PSMs: ' + str(XL))
    
    return [nXL, XL]

# re-build of percolator algorithm
def percolator(data, idColName, excludedCols, class_weight = '', I = 10, svmIter = 1000, svmC = 1.0, qTrain = 0.05, centralScoringQ = 0.01, useRankOneOnly = False, plotEveryIter = False, suppressLog = False, plotSaveName = ''):
    
    # Ugly Syntax because of ugly pandas behavior
    if (useRankOneOnly):
        df = pd.DataFrame(data[data.Rank == 1])
    else:
        df = pd.DataFrame(data)
    
    # select the scores used for learning
    scores = [x for x in list(df.columns) if (x not in excludedCols)]

    # select negative training set (yes, should use loc, not iloc)
    falseExamples = df[scores].loc[list(df[df.Label == 0].index)]
    
    # set color cycle if needed
    if(plotEveryIter):
        cols = []
        for i in range(10):
            cols.append((1-(i*0.1),0.,i*0.1))
        new_prop_cycle = cycler('color', cols)
        plt.rc('axes', prop_cycle = new_prop_cycle)

    scoreName = 'percolator_score'
    scoreNameTemp = 'temp_score'
    
   
    # iterate I times:
    for i in range(I):
        
        # split dataframe in 3. TODO?Maybe do this outside of i-for-loop?
        threeParts = np.array_split(df.sample(frac = 1, replace = False), 3)
        
        for j in [0,1,2]:
            
            # compute validation and training sets
            validate = threeParts[j]
            training = pd.concat([threeParts[k] for k in range(len(threeParts)) if(k != j)], sort = False)
                
            # compute training and response sets
            falseTrain = training.loc[list(training[training.Label == 0].index), scores]
            trueTrain = training[scores][(training['q-val'] <= qTrain) & (training.Label == 1)]
            train = falseTrain.values.tolist() + trueTrain.values.tolist()
            classes = [0] * len(falseTrain) + [1] * len(trueTrain)
            
            # set up SVM for internal cross-validation
            parameters = {'C':[0.1,1,10], 'class_weight':[{0:i, 1:1} for i in [1,3,10]]}
            W = svm.LinearSVC(dual = False, max_iter = svmIter)
            clf = GridSearchCV(W, parameters, cv = 3)
            
            # train svm with optimal C and class weights
            if(not suppressLog):
                print('Training in iteration {} with split {} starts!'.format(i + 1, j + 1))
            clf.fit(train,classes)
            
            # compute score for validation part
            X = validate[scores].values.tolist()
            validate[scoreNameTemp] = clf.decision_function(X)
            
            # merge: calculate comparable score for every PSM. TODO?Outside or inside j-for-loop?
            calcQ(validate, scoreNameTemp)
            qThreshold = min(validate[validate['q-val'] <= centralScoringQ][scoreNameTemp])
            decoyMedian = np.median(validate[validate.Label == 0][scoreNameTemp])
            validate[scoreName] = (validate[scoreNameTemp] - qThreshold) / (qThreshold - decoyMedian)
        
        df = pd.concat([validate, training])
        calcQ(df, scoreName)
        
       
        # log and plot this iteration
        if(plotEveryIter):
            pseudoROC(df, 0.05, label = 'Iteration {}'.format(i + 1))
        if(not suppressLog):
            print('Iteration {}/{} done!'.format(i + 1, I))
        
    # show plot and revert color cycle after for loop
    if(plotEveryIter):
        plt.legend(loc = 'best')
        if (plotSaveName != ''):
            plt.savefig(plotSaveName)
        plt.show()
        plt.rc('axes', prop_cycle = cycler('color', [plt.get_cmap('tab10')(i) for i in range(10)]))
    
    # WHAT TO DO WITH RANKS re-add lesser ranked PSMs to df for scoring
    #if (useRankOneOnly):
    #    df = pd.concat([df, data[data.Rank != 1]], sort = False)
    #df = addRanks(d, idColName, scoreName)
    
    # return PSMs with new score
    return df
