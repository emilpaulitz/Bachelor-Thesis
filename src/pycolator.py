import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# add columns containing FDR and the q-value respectively
def calcQ(df, scoreColName, labelColName = 'Label', isXLColName = 'NuXL:isXL', addXlQ = True):
    
    df.sort_values(scoreColName, ascending=False, inplace = True)
    df[labelColName].replace(to_replace = -1, value = 0, inplace = True)
    df['FDR'] = 1 - (df[labelColName].cumsum()/[i + 1 for i in range(len(df.index))])
    df['q-val'] = df['FDR'][::-1].cummin()[::-1]
    
    # add q-values calculated from the different classes
    if(addXlQ):
        ls = []
        for XL in [0,1]:
            
            # split the dataframe and save the subset
            currClass = pd.DataFrame(df[df[isXLColName] == XL])
            ls.append(currClass)
            
            # calculate class-specific q-value
            currClass.sort_values(scoreColName, ascending=False, inplace = True)
            FDR = 1 - (currClass[labelColName].cumsum()/[i + 1 for i in range(len(currClass))])
            currClass['class-specific_q-val'] = FDR[::-1].cummin()[::-1]
            
        df = pd.concat(ls)
        df.sort_values(scoreColName, ascending=False, inplace = True)
    return df

# add a column containing the Ranks of entries with the same id
def addRanks(df, idColName, scoreColName):
    # Sort by score to have the ranks in increasing order, and by ID to group same IDs
    df.sort_values(scoreColName, inplace = True, ascending = False)
    df.sort_values(idColName, inplace = True, kind = 'mergesort')

    # for better performance iterate over list
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

# Norm the feature columns so their values lie between 0 and 1
def normFeatures(df, excluded):
    features = [x for x in list(df.columns) if (x not in excluded)]
    for f in df[features]:
        if (max(df[f]) != min(df[f])):
            temp = df[f] - min(df[f])
            df[f] = temp / max(temp)
    return df

# Read the specified file, calculate q-values and ranks, norm the features if excluded is given and sort it according to its score
def readAndProcess(fileName, idColName,  scoreColName, excludedCols = ''):
    d = DFfromTXT(fileName)
    print('file read')
    d1 = strToNum(d)
    print('strings converted to numbers')
    d2 = calcQ(d1, scoreColName)
    print('q-values estimated')
    df = addRanks(d2, idColName, scoreColName)
    print('ranks computed')
    if (not excludedCols == ''):
        df = normFeatures(df, excludedCols)
        print('features normed')
    df.sort_values(scoreColName, inplace = True, ascending = False)
    print('file ready')
    return df   

# Plot Pseudo ROC of the df including entries with q in x = [0,xMax] and return area under the curve
def pseudoROC(df, xMax = 0.05, onlyFirstRank = True, onlyVals = False, qColName = 'q-val', rankColName = 'Rank', title = '', label = ''):
    if (onlyFirstRank):
        qVals = df[(df[qColName] <= xMax) & (df[rankColName] == 1)][qColName].values.tolist()
    else: 
        qVals = df[df[qColName] <= xMax][qColName].values.tolist()
    qVals.sort()
    if (onlyVals):
        return qVals
    plt.xlim(0,xMax)
    plt.ylim(0,len(qVals))
    plt.title(title)
    AUC = auc(qVals, range(len(qVals)))
    if (label != ''):
        label += ', '
    plt.plot(qVals, range(len(qVals)), label = label + 'AUC: ' + str(round(AUC, 2)))
    plt.legend(loc = 'best')
    return AUC

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
def evalXL(df, qColName = 'class-specific_q-val'):
    nXLauc = pseudoROC(df[df['NuXL:isXL'] == 0], qColName = qColName, label = 'not cross-linked')
    XLauc = pseudoROC(df[df['NuXL:isXL'] == 1], qColName = qColName, label = 'cross-linked')
    plt.legend(loc = 'best')    
    return [nXLauc, XLauc]

# plot pseudo ROC for every iteration (see percolator)
def pseudoROCiter(plotList, I, nameList, plotSaveName, plotDPI):
    cols = []
    for i in range(I):
        cols.append((1-(i*(1/I)),0.,i*(1/I)))
    new_prop_cycle = cycler('color', cols)
    plt.rc('axes', prop_cycle = new_prop_cycle)
        
    for i in [0,1,2]:
        plt.xlim(0,0.05)
        plt.ylim(0,max([len(plotList[i][j]) for j in range(I)]))
        plt.title('Pseudo ROC-Curve of {} PSMs'.format(nameList[i]))
        for j in range(I):
            currAuc = round(auc(plotList[i][j], range(len(plotList[i][j]))), 2)
            plt.plot(plotList[i][j], range(len(plotList[i][j])), label = 'Iteration {}, AUC: {}'.format(j + 1, currAuc))
        plt.legend(loc = 'best')
        if (plotSaveName != ''):
            if ('{}' in plotSaveName):
                name = plotSaveName.format(nameList[i])
            else:
                name = nameList[i] + plotSaveName
            plt.savefig(name, dpi = plotDPI)
        plt.show()
    
    # reset color cycle
    plt.rc('axes', prop_cycle = cycler('color', [plt.get_cmap('tab10')(i) for i in range(10)]))
    
# End for loop by setting the scores to the best iteration (given by bestIterReverse)
def percEndFor(df, scoreName, idColName, bestIterReverse, scoresIter):
    df.sort_index(inplace = True)
    df[scoreName] = scoresIter[-bestIterReverse]
    df = calcQ(df, scoreName)
    df = addRanks(df, idColName, scoreName)

# re-build of percolator algorithm.
def percolator(data, idColName, excludedCols, I = 10, svmIter = 1000, svmC = 1.0, qTrain = 0.05, centralScoringQ = 0.01, useRankOneOnly = False, plotEveryIter = True, suppressLog = False, plotSaveName = '', plotDPI = 100, termWorseIters = 4, rankOption = True):
    
    # Ugly Syntax because of ugly pandas behavior
    if (useRankOneOnly):
        df = pd.DataFrame(data[data.Rank == 1])
    else:
        df = pd.DataFrame(data)
    
    scores = [x for x in list(df.columns) if (x not in excludedCols)]
    if(plotEveryIter):
        plotList = [[],[],[]]
    scoreName = 'percolator_score'
    scoreNameTemp = 'temp_score'
    aucIter = []
    scoresIter = []
    
    for i in range(I):
        
        # split dataframe in 3. TODO?Maybe do this outside of i-for-loop?
        threeParts = np.array_split(df.sample(frac = 1, replace = False), 3)
        
        for j in [0,1,2]:
            
            validate = threeParts[j]
            training = pd.concat([threeParts[k] for k in range(3) if(k != j)], sort = False)
            
            # calc SpecIds, of which the first rank is a decoy and include corresponding PSMs in neg train set
            if(rankOption):
                badSpecs = training[(training['Rank'] == 1) & (training['Label'] == 0)][idColName].tolist()
                goodSpecs = list(set(training[idColName]) - set(badSpecs))
            
                # compute training and response sets (yes, should use loc, not iloc)
                falseTrain = training.loc[list(training[(training.Label == 0) | (training[idColName].isin(badSpecs))].index), scores]
                trueTrain = training.loc[list(training[(training['q-val'] <= qTrain) & (training.Label == 1) & (training[idColName].isin(goodSpecs))].index), scores]
            else:
                falseTrain = training.loc[list(training[training.Label == 0].index), scores]
                trueTrain = training.loc[list(training[(training['q-val'] <= qTrain) & (training.Label == 1)].index), scores]
            train = falseTrain.values.tolist() + trueTrain.values.tolist()
            classes = [0] * len(falseTrain) + [1] * len(trueTrain)
            
            # set up SVM using internal cross-validation
            parameters = {'C':[0.1,1,10], 'class_weight':[{0:i, 1:1} for i in [1,3,10]]}
            W = svm.LinearSVC(dual = False, max_iter = svmIter)
            clf = GridSearchCV(W, parameters, cv = 3)
            if(not suppressLog):
                print('Training in iteration {} with split {}/3 starts!'.format(i + 1, j + 1))
            clf.fit(train,classes)
            
            # compute score for validation part
            X = validate[scores].values.tolist()
            validate[scoreNameTemp] = clf.decision_function(X)
            
            # merge: calculate comparable score
            calcQ(validate, scoreNameTemp, addXlQ = False)
            qThreshold = min(validate[validate['q-val'] <= centralScoringQ][scoreNameTemp])
            decoyMedian = np.median(validate[validate.Label == 0][scoreNameTemp])
            validate[scoreName] = (validate[scoreNameTemp] - qThreshold) / (qThreshold - decoyMedian)
        
        # merge the three parts and calculate q based on comparable score
        df = pd.concat([validate, training])
        df = calcQ(df, scoreName)
        df = addRanks(df, idColName, scoreName)
        
        # plot and calc auc for this iteration
        if(plotEveryIter):
            for plot in [0,1]:
                plotList[plot].append(pseudoROC(df[df['NuXL:isXL'] == plot], onlyVals = True, qColName = 'class-specific_q-val'))
            plotList[2].append(pseudoROC(df, onlyVals = True))
            x = plotList[2][i]
        else:
            x = pseudoROC(df, onlyVals = True)
        aucIter.append(auc(x, range(len(x))))
        df.sort_index(inplace = True)
        scoresIter.append(df[scoreName])
        
        # Terminate if auc has not been getting better in last termWorseIters iterations
        if(i + 1 >= termWorseIters):
            if(aucIter[-termWorseIters] == max(aucIter)):
                if(not suppressLog):
                    print('Results are not getting better. Terminating and using Iteration {} with an auc of {}.'.format(i + 2 - termWorseIters, round(max(aucIter),2)))
                I = i + 1
                percEndFor(df, scoreName, idColName, termWorseIters, scoresIter)
                break
        
        if(not suppressLog):
            print('Iteration {}/{} done!'.format(i + 1, I))
            
        # in the end, choose the best of the last iterations
        if(i + 1 == I):
            for j in range(2,termWorseIters):
                if(aucIter[-j] == max(aucIter)):
                    percEndFor(df, scoreName, idColName, j, scoresIter)
                    print('Terminating and using Iteration {} with an auc of {}.'.format(i + 2 - j, round(max(aucIter),2)))
            
    result = pd.DataFrame(df[df.Rank == 1])
    result = calcQ(result, scoreName)
    
    # generate plots and revert color cycle after calculations are done
    if(plotEveryIter):
        pseudoROCiter(plotList, I, ['non-XL', 'XL', 'all'], plotSaveName, plotDPI)
    
    # return PSMs with new score
    return result
