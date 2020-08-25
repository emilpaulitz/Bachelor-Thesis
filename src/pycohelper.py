import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.utils import shuffle
from cycler import cycler
from itertools import product

# Read text file and turn it into a pandas data frame
def DFfromTXT(fileName):
    with open(fileName, 'r') as file:
        txt = file.read()

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
def strToFloat(df):
    for col in df:
        try:
            df[col] = [float(i) for i in df[col]]
        except ValueError:
            continue
    return df

# add columns containing FDR and the q-value respectively
def calcQ(df, scoreColName, labelColName = 'Label', isXLColName = 'NuXL:isXL', addXlQ = True, ascending = False):

    df.sort_values(scoreColName, ascending=ascending, inplace = True)
    df[labelColName].replace(to_replace = -1, value = 0, inplace = True)
    df['FDR'] = 1 - (df[labelColName].cumsum()/range(1, len(df) + 1))
#     df['FDR'] = (range(1, len(df) + 1)/df[labelColName].cumsum()) - 1
    df['q-val'] = df['FDR'][::-1].cummin()[::-1]
    
    # add q-values calculated from the different classes
    if(addXlQ):
        ls = []
        for XL in [0,1]:
            
            # split the dataframe and save the subset
            currClass = pd.DataFrame(df[df[isXLColName] == XL])
            ls.append(currClass)
            
            # calculate class-specific q-value
            currClass.sort_values(scoreColName, ascending=ascending, inplace = True)
            FDR = 1 - (currClass[labelColName].cumsum()/range(1, len(currClass) + 1))
            currClass['class-specific_q-val'] = FDR[::-1].cummin()[::-1]
            
        df = pd.concat(ls)
        df.sort_values(scoreColName, ascending=ascending, inplace = True)
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

# norm the feature columns so their values lie between 0 and 1 and throw out meaningless columns
def normFeatures(df, features):
    for f in df[features]:
        if (max(df[f]) != min(df[f])):
            temp = df[f] - min(df[f])
            df[f] = temp / max(temp)
        else:
            df.drop(columns = f, inplace = True)
    return df

# convert every col into an int if possible
def floatToInt(df):
    for col in df:
        try:
            if all([i.is_integer() for i in df[col]]):
                df[col] = [int(x) for x in df[col]]
        except:
            continue
    return df

# split data in three parts by selected method (experimental)
def percSplitOuter(df, scanNrTest, peptideTest, balancingOuter, propTarDec, propXLnXL):
    if (scanNrTest):
        # ScanNrs together tests
        subgroups = np.array_split(df['ScanNr'].sample(frac=1, replace = False).unique(), 3)
        return [pd.DataFrame(df.loc[df['ScanNr'].isin(subgroups[part])]) for part in range(3)]
    
    elif (peptideTest):
        # Same peptides together tests (why is the code structure different?)
        peptides = pd.Series(df['Peptide'].unique())
        subgroups = np.array_split(shuffle(peptides), 3)
        return [pd.DataFrame(df.loc[df['Peptide'].isin(subgroups[part])]) for part in range(3)]
    
    elif (balancingOuter and (propTarDec or propXLnXL)):
        
        if (propTarDec and propXLnXL):
            comb = [df.loc[(df.Label == a) & (df['NuXL:isXL'] == b)] for a,b in product([0,1], repeat = 2)]
            
        elif (propTarDec):
            comb = [df.loc[df.Label == a] for a in [0,1]]
            
        elif (propXLnXL):
            comb = [df.loc[df['NuXL:isXL'] == b] for b in [0,1]]
            
        genSplits = map(lambda part: np.array_split(shuffle(part), 3), comb)
        return [pd.concat(z) for z in zip(*genSplits)]
    
    return np.array_split(shuffle(df), 3)

# select negative and positive training set using given method (experimental)
def percSelectTrain(training, qTrain, rankOption, lowRankDecoy, idColName):
    if(rankOption):
        badSpecs = training.loc[(training['Rank'] == 1) & (training['Label'] == 0), idColName].tolist()

        # compute training and response sets (yes, should use loc, not iloc)
        falseTrain = training.loc[(training.Label == 0) | (training[idColName].isin(badSpecs))]
        trueTrain = training.loc[(training['q-val'] <= qTrain) & (training.Label == 1) & (~training[idColName].isin(badSpecs))]
    elif (lowRankDecoy):
        falseTrain = training.loc[(training.Label == 0) | (training.Rank > 1)]
        trueTrain = training.loc[(training['q-val'] <= qTrain) & (training.Label == 1) & (training.Rank == 1)]
    else:
        falseTrain = training.loc[training.Label == 0]
        trueTrain = training.loc[(training['q-val'] <= qTrain) & (training.Label == 1)]
        
    return falseTrain, trueTrain

# calculate the features that are 0 for ever non-cross-linked PSM
def percPrepImputation(df, xlcol):
    ls = []
    for col in df.columns:
        ls.append([col])
        ls[-1].append(min(df[col]))
        ls[-1].append(max(df[col]))
        ls[-1].append(min(df.loc[df[xlcol] == 1, col]))
        ls[-1].append(max(df.loc[df[xlcol] == 1, col]))
        ls[-1].append(min(df.loc[df[xlcol] == 0, col]))
        ls[-1].append(max(df.loc[df[xlcol] == 0, col]))

    temp = pd.DataFrame(ls, columns = ['Col','Min Ges', 'Max Ges', 'Min XL', 'Max XL', 'Min nXL', 'Max nXL'])
    colsToImputate = list(temp.loc[(temp['Min nXL'] == temp['Max nXL']) & (temp['Min XL'] != temp['Max XL']), 'Col'])
    if(all(df.loc[df[xlcol] == 0, colsToImputate] == 0)):
        df.loc[df[xlcol] == 0, colsToImputate] = np.nan
    else:
        print('reexamine colsToImputate')
    return colsToImputate

# initalize classifier using selected option (experimental)
def percInitClf(falseTrain, trueTrain, train, classes, balancedOption, KFoldTest, scanNrTest, peptideTest, fastCV, propTarDec, propXLnXL, balancingInner, specialGrid):
    if (fastCV):
        parameters = {'C':[1,10], 'class_weight':[{0:1, 1:1}]}
    elif (specialGrid):        
        parameters = {'C':[0.1,1,10], 'class_weight':[{0:i, 1:10} for i in [3,10,30]]}
    else:
        parameters = {'C':[0.1,1,10], 'class_weight':[{0:i, 1:1} for i in [1,3,10]]}
    if(balancedOption):
        parameters['class_weight'] += ['balanced']
    W = svm.LinearSVC(dual = False)
    
    if(KFoldTest):
        # use normal kfold instead of stratified kfold
        clf = GridSearchCV(W, parameters, cv = KFold(n_splits=3, shuffle=True))
    elif(balancingInner and (propTarDec or propXLnXL)):
        
        df = pd.concat([falseTrain,trueTrain])
        
        # balance subsets of inner cv in order to keep the proportions of tar/dec or XL/nXL the same as in the whole dataset
        if(propTarDec and propXLnXL):
            comb = [df.loc[(df.Label == a) & (df['NuXL:isXL'] == b)] for a,b in product([0,1], repeat = 2)]
        if(propTarDec):
            comb = [df.loc[df.Label == a] for a in [0,1]]
        if(propXLnXL):
            comb = [df.loc[df['NuXL:isXL'] == b] for b in [0,1]]
            
        # Generate a list of lists of indices, with the correct proportions in every sublist.
        genSplits = map(lambda part: np.array_split(shuffle(part.index), 3), comb)

        # zip the lists and flatten them
        indices = [[item for sublist in z for item in sublist] for z in zip(*genSplits)]
            
        # Convert this nested list into a Series, containing the corresponding split in every index
        groups = pd.Series(index = df.index)
        for i in [0,1,2]:
            groups.loc[groups.index.isin(indices[i])] = i
            
        clf = GridSearchCV(W, parameters, cv = GroupKFold(n_splits=3).split(train, classes, groups))
    elif(scanNrTest):
        groups = falseTrain['ScanNr'].tolist() + trueTrain['ScanNr'].tolist()
        clf = GridSearchCV(W, parameters, cv = GroupKFold(n_splits=3).split(train, classes, groups))
    elif(peptideTest):
        groups = falseTrain['Peptide'].tolist() + trueTrain['Peptide'].tolist()
        clf = GridSearchCV(W, parameters, cv = GroupKFold(n_splits=3).split(train, classes, groups))
    else:
        clf = GridSearchCV(W, parameters, cv = 3)
    return clf
    
# End for loop by setting the scores to the best iteration (given by bestIterReverse)
def percEndFor(df, scoreName, idColName, bestIter):
    df[scoreName] = df['scoresIter_{}'.format(bestIter)]
    cols = df.columns
    df.drop([cols[i] for i in range(len(cols)) if(cols[i].startswith('scoresIter_'))], axis = 1, inplace = True)
    df = calcQ(df, scoreName)
    df = addRanks(df, idColName, scoreName)
    
# plot pseudo ROC for every iteration (see percolator)
def pseudoROCiter(plotList, I, nameList, plotSaveName, plotDPI, plotXLnXL, identsAsMetric):
        
    # determine over which parts of the plotList to iterate
    if (plotXLnXL and len(nameList) == 3):
        iterate = [0,1,2]
    else:
        iterate = [2]
        
    if (identsAsMetric):
        for i in iterate:
            plt.ylim(0,max([max(plotList[plot]) for plot in iterate]) * 1.05)
            plt.plot(range(1,I + 1), plotList[i], label = nameList[i])
            plt.legend(loc = 'best')
        plt.title('Number of identified PSMs at q = 0.01'.format(nameList[i]))
        if (plotSaveName != ''):
            plt.savefig(plotSaveName, dpi = plotDPI)
        plt.show()
    else:
        # set new color cycle to a gradient from red to blue
        cols = []
        for i in range(I):
            cols.append(( 1-(i/(I-1)), 0., i/(I-1) ))
        new_prop_cycle = cycler('color', cols)
        plt.rc('axes', prop_cycle = new_prop_cycle)
        
        for i in iterate:
            plt.xlim(0,0.05)
            plt.ylim(0,max([len(plotList[i][j]) for j in range(I)]))
            plt.title('Pseudo ROC-Curve of {} PSMs'.format(nameList[i]))

            # generate pseudo ROC for every iteration represented in plotList
            for j in range(I):
                currAuc = round(auc(plotList[i][j], range(len(plotList[i][j]))), 2)
                plt.plot(plotList[i][j], range(len(plotList[i][j])), label = 'Iteration {}, AUC: {}'.format(j + 1, currAuc))

            # generate legend and save figure if specified
            plt.legend(loc = 'best')
            if (plotSaveName != ''):
                if ('{}' in plotSaveName):
                    name = plotSaveName.format(nameList[i])
                else:
                    lastDot = plotSaveName.rfind('.')
                    name = plotSaveName[:lastDot] + '_' + nameList[i] + plotSaveName[lastDot:]
                plt.savefig(name, dpi = plotDPI)
            plt.show()

        # reset color cycle to standard
        plt.rc('axes', prop_cycle = cycler('color', [plt.get_cmap('tab10')(i) for i in range(10)]))