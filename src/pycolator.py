import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, GroupKFold, LeaveOneGroupOut
from cycler import cycler
from itertools import product
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pycohelper import *

# Read the specified file, calculate q-values and ranks, norm the features if excluded is given and sort it according to its score
def readAndProcess(fileName, idColName,  scoreColName, excludedCols = ''):
    d = DFfromTXT(fileName)
    print('file read...')
    d1 = strToFloat(d)
    print('strings converted to floats...')
    d2 = calcQ(d1, scoreColName)
    print('q-values estimated...')
    df = addRanks(d2, idColName, scoreColName)
    print('ranks computed...')
    if (not excludedCols == ''):
        df = normFeatures(df, [x for x in list(df.columns) if (x not in excludedCols)])
        print('features normed...')
        df = floatToInt(df)
        print('floats converted to ints...')
    df.sort_values(scoreColName, inplace = True, ascending = False)
    print('file ready!')
    return df   

# Plot Pseudo ROC of the df including entries with q in x = [0,xMax] and return area under the curve
def pseudoROC(df, xMax = 0.05, onlyFirstRank = True, onlyVals = False, qColName = 'q-val', rankColName = 'Rank', title = '', label = '', plot = True):
    if (onlyFirstRank):
        temp = pd.DataFrame(df.loc[df.Rank == 1, [qColName, 'Label', rankColName, 'NuXL:isXL']])
        temp = calcQ(temp, qColName, ascending = True)
        qVals = temp.loc[temp[qColName] <= xMax, qColName].values.tolist()
    else: 
        xMax = float(xMax)
        qVals = df.loc[df[qColName] <= xMax, qColName].values.tolist()
    qVals.sort()
    if (onlyVals):
        return qVals
    if (not plot):
        return auc(qVals, range(len(qVals)))
    plt.xlim(-0.001,xMax)
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
    # fill with nan to bring all lists to the same length
    l = max([len(ls) for ls in lss])
    for ls in lss:
        ls += [float('NaN')] * (l - len(ls))
        
    # calculate best, worst and median outcomes at every point in the graph
    mx = [min(elem) for elem in zip(*lss)]
    mn = [max(elem) for elem in zip(*lss)]
    md = [np.median(elem) for elem in zip(*lss)]
    
    # plot them
    plt.xlim(0, xMax)
    plt.ylim(0, len([x for x in mx if(x <= xMax)]))
    plt.plot(mx, range(len(mx)), label = labels[0])
    plt.plot(mn, range(len(mn)), label = labels[1])
    plt.plot(md, range(len(md)), label = labels[2])
    plt.legend(loc = 'best')
    
# plot pseudoROCs for XLs and Non XLs and return the areas under the curves
def evalXL(df, qColName = 'class-specific_q-val', plot = True):
    nXLauc = pseudoROC(df[df['NuXL:isXL'] == 0], qColName = qColName, plot = plot, label = 'not cross-linked')
    XLauc = pseudoROC(df[df['NuXL:isXL'] == 1], qColName = qColName, plot = plot, label = 'cross-linked')
    return (nXLauc, XLauc)

# percolator with several experimental options
def percolator_experimental(df, idColName, features, I = 10, qTrain = 0.05, centralScoringQ = 0.01, useRankOneOnly = False, plotEveryIter = True, plotXLnXL = True, suppressLog = False, plotSaveName = '', plotDPI = 100, termWorseIters = 4, cutOffImprove = 0.01, rankOption = False, scanNrTest = False, peptideTest = False, lowRankDecoy = False, KFoldTest = False, balancedOption = False, fastCV = False, propTarDec = True, propXLnXL = True, balancingInner = True, balancingOuter = True, optimalRanking = True, specialGrid = True, identsAsMetric = 'automatic', multImputation = False):

    if (useRankOneOnly):
        df = df.loc[df.Rank == 1]
    
    if(plotEveryIter):
        plotList = [[],[],[]]
    scoreName = 'percolator_score'
    scoreNameTemp = 'temp_score'
    metricIter = []
    scoresIter = []
    
    for i in range(I):
        
        # If the results are improving by less than cutOffImprove, cut off every entry with a worse rank than 1
        if (optimalRanking and (i >= 2) and metricIter[-1] < (metricIter[-2]*(1 + cutOffImprove))):
            if (not suppressLog):
                print('Re-ranking complete. Cutting off low ranks.\n')
            df = df.loc[df.Rank == 1]
            optimalRanking = False
            
        threeParts = percSplitOuter(df, scanNrTest, peptideTest, balancingOuter, propTarDec, propXLnXL)
        
        for j in [0,1,2]:
            
            validate = threeParts[j]
            training = pd.concat([threeParts[k] for k in range(3) if(k != j)], sort = False)
                        
            # calc SpecIds, of which the first rank is a decoy and include corresponding PSMs in neg train set
            falseTrain, trueTrain = percSelectTrain(training, qTrain, rankOption, lowRankDecoy, idColName)
            if(len(trueTrain) < 3 or len(falseTrain) < 3):
                raise ValueError('Dataset too small. There are not enough positive or negative examples to perform nested cross-validation.')
            
            train = falseTrain[features].values.tolist() + trueTrain[features].values.tolist()
            classes = [0] * len(falseTrain) + [1] * len(trueTrain)
            
            # imputation
            if (multImputation):
                colsToImputate = percPrepImputation(df, 'NuXL:isXL')
                if (not all(df.loc[df['NuXL:isXL'] == 0, colsToImputate] == np.nan)):
                    print('error in imputation')
                imp = IterativeImputer()#estimator = svm.LinearSVC(), max_iter=10)
                imp.fit(train)
                training[features] = imp.transform(training[features])
                validate[features] = imp.transform(validate[features])
                
                falseTrain, trueTrain = percSelectTrain(training, qTrain, rankOption, lowRankDecoy, idColName)
                train = falseTrain[features].values.tolist() + trueTrain[features].values.tolist()
                classes = [0] * len(falseTrain) + [1] * len(trueTrain)
            
            # set up SVM using internal cross-validation
            clf = percInitClf(falseTrain, trueTrain, train, classes, balancedOption, KFoldTest, scanNrTest, peptideTest, fastCV, propTarDec, propXLnXL, balancingInner, specialGrid)
            
            if(not suppressLog):
                print('Training in iteration {} with split {}/3 starts!'.format(i + 1, j + 1))
                print('Length of positive trainingset: {}, length of negative training set: {}'.format(len(trueTrain), len(falseTrain)))
            clf.fit(train,classes)
            if(not suppressLog):
                print('Optimal parameters are C={} and class_weight={}.\n'.format(clf.best_params_['C'], clf.best_params_['class_weight']))
            
            # compute score for validation part
            X = validate[features].values.tolist()
            validate[scoreNameTemp] = clf.decision_function(X)
            
            # merge: calculate comparable score
            calcQ(validate, scoreNameTemp, addXlQ = False)
            try:
                qThreshold = min(validate.loc[validate['q-val'] <= centralScoringQ, scoreNameTemp])
            except:
                raise ValueError('Dataset too small. Score normalization not possible, increase centralScoringQ.')
            decoyMedian = np.median(validate.loc[validate.Label == 0, scoreNameTemp])
            validate[scoreName] = (validate[scoreNameTemp] - qThreshold) / (qThreshold - decoyMedian)
        
        # merge the three parts and calculate q based on comparable score
        df = pd.concat([validate, training])
        df = calcQ(df, scoreName)
        df = addRanks(df, idColName, scoreName)
        
        # determine metric to use
        if (identsAsMetric == 'automatic'):
            identsAsMetric = len(df.loc[(df.Rank == 1) & (df['q-val'] > 0) & (df['q-val'] <= 0.05)]) == 0
            if(not suppressLog):
                if (identsAsMetric):
                    print('Using identifications at 1% q-value with rank 1 as metric.')
                else:
                    print('Using AUC of pseudo-ROC with x = [0,0.05] as metric.')

        if (identsAsMetric):
            metric = 'identifications'
        else:
            metric = 'an auc'
        
        # plot metric for this iteration
        if(plotEveryIter):
            if(identsAsMetric):
                if (plotXLnXL):
                    for plot in [0,1]:
                        plotList[plot].append(len(df.loc[(df.Rank == 1) & (df['class-specific_q-val'] <= 0.01) & (df['NuXL:isXL'] == plot)]))
                plotList[2].append(len(df.loc[(df.Rank == 1) & (df['q-val'] <= 0.01)]))
            else:
                if (plotXLnXL):
                    for plot in [0,1]:
                        plotList[plot].append(pseudoROC(df.loc[df['NuXL:isXL'] == plot], onlyVals = True, qColName = 'class-specific_q-val'))
                plotList[2].append(pseudoROC(df, onlyVals = True))              
            
        # Calc metric for this iteration
        if (identsAsMetric):
            metricIter.append(len(df.loc[(df.Rank == 1) & (df['q-val'] <= 0.01)]))
        else:
            metricIter.append(pseudoROC(df, plot = False))
        df['scoresIter_{}'.format(i)] = df[scoreName]
        
        if(not suppressLog):
            print('Iteration {}/{} done! It yielded {} of {}.\n'.format(i + 1, I, metric, metricIter[-1]))
            
        # Terminate if metric has not been getting better in last termWorseIters iterations
        if((i + 1 >= termWorseIters) and (metricIter[-termWorseIters] == max(metricIter))):
            I = i + 1
            percEndFor(df, scoreName, idColName, I - termWorseIters)
            if(not suppressLog):
                print('Results are not getting better. Terminating and using Iteration {} with {} of {}.'.format(i + 2 - termWorseIters, metric, round(max(metricIter),2)))
            break
        
        # in the end, choose the best of the last iterations
        if((i + 1 == I) and (i + 1 >= termWorseIters)):
            for j in range(1,termWorseIters):
                if(metricIter[-j] == max(metricIter)):
                    percEndFor(df, scoreName, idColName, I - j)
                    if(not suppressLog):
                        print('Terminating and using Iteration {} with {} of {}.'.format(i + 2 - j, metric, round(max(metricIter),2)))
            
    df = df.loc[df.Rank == 1]
    df = calcQ(df, scoreName)
    
    # generate plots and revert color cycle after calculations are done
    if(plotEveryIter):
        pseudoROCiter(plotList, I, ['non-XL', 'XL', 'all'], plotSaveName, plotDPI, plotXLnXL, identsAsMetric)
    
    # return PSMs with new score
    return df

# clean re-build of percolator algorithm.
# not up to date!
def percolator(df, idColName, features, I = 10, qTrain = 0.05, centralScoringQ = 0.01, plotEveryIter = True, suppressLog = False, plotSaveName = '', plotDPI = 100, termWorseIters = 4):
    
    if(plotEveryIter):
        plotList = [[],[],[]]
    scoreName = 'percolator_score'
    scoreNameTemp = 'temp_score'
    aucIter = []
    scoresIter = []
    
    for i in range(I):         
        
        # split dataframe in 3.
        threeParts = np.array_split(df.sample(frac = 1, replace = False), 3)
        
        for j in [0,1,2]:
            
            validate = threeParts[j]
            training = pd.concat([threeParts[k] for k in range(3) if(k != j)], sort = False)
            
            # calc SpecIds, of which the first rank is a decoy and include corresponding PSMs in neg train set
            badSpecs = training.loc[(training['Rank'] == 1) & (training['Label'] == 0), idColName].tolist()

            # compute training and response sets (yes, should use loc, not iloc)
            falseTrain = training.loc[(training.Label == 0) | (training[idColName].isin(badSpecs)), features]
            trueTrain = training.loc[(training['q-val'] <= qTrain) & (training.Label == 1) & (~training[idColName].isin(badSpecs)), features]
            train = falseTrain.values.tolist() + trueTrain.values.tolist()
            classes = [0] * len(falseTrain) + [1] * len(trueTrain)
            
            # set up SVM using internal cross-validation
            parameters = {'C':[0.1,1,10], 'class_weight':[{0:i, 1:1} for i in [1,3,10]]}
            W = svm.LinearSVC(dual = False)
            clf = GridSearchCV(W, parameters, cv = 3)
            
            if(not suppressLog):
                print('Training in iteration {} with split {}/3 starts!'.format(i + 1, j + 1))
            clf.fit(train,classes)
            
            # compute score for validation part
            X = validate[features].values.tolist()
            validate[scoreNameTemp] = clf.decision_function(X)
            
            # merge: calculate comparable score
            calcQ(validate, scoreNameTemp, addXlQ = False)
            qThreshold = min(validate.loc[validate['q-val'] <= centralScoringQ, scoreNameTemp])
            decoyMedian = np.median(validate.loc[validate.Label == 0, scoreNameTemp])
            validate[scoreName] = (validate[scoreNameTemp] - qThreshold) / (qThreshold - decoyMedian)
        
        # merge the three parts and calculate q based on comparable score
        df = pd.concat([validate, training])
        df = calcQ(df, scoreName)
        df = addRanks(df, idColName, scoreName)
        
        # plot and calc auc for this iteration
        if(plotEveryIter):
            for plot in [0,1]:
                plotList[plot].append(pseudoROC(df.loc[df['NuXL:isXL'] == plot], onlyVals = True, qColName = 'class-specific_q-val'))
            plotList[2].append(pseudoROC(df, onlyVals = True))
            x = plotList[2][i]
        else:
            x = pseudoROC(df, onlyVals = True)
        aucIter.append(auc(x, range(len(x))))
        df['scoresIter_{}'.format(i)] = df[scoreName]
        
        # Terminate if auc has not been getting better in last termWorseIters iterations
        if(i + 1 >= termWorseIters):
            if(aucIter[-termWorseIters] == max(aucIter)):
                I = i + 1
                percEndFor(df, scoreName, idColName, termWorseIters, I)
                if(not suppressLog):
                    print('Results are not getting better. Terminating and using Iteration {} with an auc of {}.'.format(i + 2 - termWorseIters, round(max(aucIter),2)))
                break
        
        if(not suppressLog):
            print('Iteration {}/{} done!'.format(i + 1, I))
            
        # in the end, choose the best of the last iterations
        if((i + 1 == I) and (i + 1 >= termWorseIters)):
            for j in range(1,termWorseIters):
                if(aucIter[-j] == max(aucIter)):
                    percEndFor(df, scoreName, idColName, j, I)
                    print('Terminating and using Iteration {} with an auc of {}.'.format(i + 2 - j, round(max(aucIter),2)))
            
    df = df.loc[df.Rank == 1]
    df = calcQ(df, scoreName)
    
    # generate plots and revert color cycle after calculations are done
    if(plotEveryIter):
        pseudoROCiter(plotList, I, ['non-XL', 'XL', 'all'], plotSaveName, plotDPI)
    
    # return PSMs with new score
    return df
