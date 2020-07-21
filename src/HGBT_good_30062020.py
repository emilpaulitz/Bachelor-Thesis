#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pyopenms as oms
import sys
from sklearn.model_selection import train_test_split
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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
    
    if (ascending):
        df.sort_values(scoreColName, ascending=True, inplace = True)
    else:
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

# norm the feature columns so their values lie between 0 and 1 and throw out meaningless columns
def normFeatures(df, features):
    for f in df[features]:
        if (max(df[f]) != min(df[f])):
            try:
                temp = df[f] - min(df[f])
            except:
                print("Error: Column not of float type.")
                print(f)
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

# Read the specified file, calculate q-values and ranks, norm the features if excluded is given and sort it according to its score
def readAndProcessTSV(fileName, idColName,  scoreColName, excludedCols = '', normalizeCols=False):
    d = DFfromTXT(fileName)
    print('file read...')
    d1 = strToFloat(d)
    print('strings converted to floats...')
    d2 = calcQ(d1, scoreColName)
    print('q-values estimated...')
    df = addRanks(d2, idColName, scoreColName)
    print('ranks computed...')
    if (not excludedCols == ''):
        includedCols = [x for x in list(df.columns) if (x not in excludedCols)]
        if normalizeCols:
            print("Normalizing columns:")
            print(includedCols)
            df = normFeatures(df, includedCols)
        print('features normed...')
        df = floatToInt(df)
        print('floats converted to int...')
    df.sort_values(scoreColName, inplace = True, ascending = False)
    print('file ready!')
    return df   

def readAndProcessIdXML(input_file, idColName,  scoreColName, excludedCols = '', normalizeCols=False):
    from pyopenms import IdXMLFile
    prot_ids = []; pep_ids = []
    IdXMLFile().load(input_file, prot_ids, pep_ids)
    meta_value_keys = []
    rows = []
    for peptide_id in pep_ids:
        spectrum_id = peptide_id.getMetaValue("spectrum_reference")  
        scan_nr = spectrum_id[spectrum_id.rfind('=') + 1 : ]

        hits = peptide_id.getHits()

        psm_index = 0
        for h in hits:
            charge = h.getCharge()
            z2 = 0; z3 = 0; z4 = 0; z5 = 0

            if charge == 2:
                z2 = 1
            if charge == 3:
                z3 = 1
            if charge == 4:
                z4 = 1
            if charge == 5:
                z5 = 1
#            rank = h.getRank()
 #           score = h.getScore()
            if "target" in h.getMetaValue("target_decoy"):
                label = 1
            else:
                label = 0
            sequence = h.getSequence().toString()

            if len(meta_value_keys) == 0:
                h.getKeys(meta_value_keys)
                meta_value_keys = [x.decode() for x in meta_value_keys if not ("target_decoy" in x.decode() or "spectrum_reference" in x.decode() or "rank" in x.decode() or x.decode() in excludedCols)]
                all_columns = ['SpecId','PSMId','Label','ScanNr','Peptide','peplen','ExpMass','charge2','charge3','charge4','charge5'] + meta_value_keys
                print(all_columns)
            # static part
            row = [spectrum_id, psm_index, label, scan_nr, sequence, str(len(sequence)), peptide_id.getMZ(), z2, z3, z4, z5]
            # scores in meta values
            for k in meta_value_keys:
                if not ("target_decoy" in k or "spectrum_reference" in k or "rank" in k or k in excludedCols): # don't add them twice
                    s = h.getMetaValue(k)
                    if type(s) == bytes:
                        s = s.decode()
                    row.append(s)
            rows.append(row)
            psm_index += 1
    d = pd.DataFrame(rows, columns=all_columns)

    print (d.head())

    print('Converting column types')
    d1 = strToFloat(d)
    print('Calculating q-values')
    d2 = calcQ(d1, scoreColName)
    print('Assigning ranks')
    df = addRanks(d2, idColName, scoreColName)
    
    if (not excludedCols == ''):
        includedCols = [x for x in list(df.columns) if (x not in excludedCols)]
        if normalizeCols:
            print("Normalizing columns:")
            print(includedCols)
            df = normFeatures(df, includedCols)
        print('features normed...')
        df = floatToInt(df)
        print('floats converted to int...')
    df.sort_values(scoreColName, inplace = True, ascending = False)
    print('file ready!')
    return df   


def storeIdXML(input_file, df, output_file):
    map_id2pvalue = dict()
    for index, row in df.iterrows():
        key = row["SpecId"] + "_" + str(row["PSMId"])
        #map_id2pvalue[key] = row["p-value"]
        map_id2pvalue[key] = row["class-specific_q-val"]

    # load old data again
    from pyopenms import IdXMLFile, PeptideHit
    prot_ids = []; pep_ids = []
    IdXMLFile().load(input_file, prot_ids, pep_ids)

    new_pep_ids = []
    for peptide_id in pep_ids:
        hits = peptide_id.getHits()
        psmid = 0
        specid = peptide_id.getMetaValue("spectrum_reference").decode()
        new_hits = []
        for h in hits:
            key = specid + "_" + str(psmid)
            if key in map_id2pvalue:
                h.setScore(map_id2pvalue[key])
                new_hits.append(h)
            psmid += 1
        peptide_id.setHits(new_hits)
        peptide_id.setScoreType("probability")
        new_pep_ids.append(peptide_id)

    IdXMLFile().store(output_file, prot_ids, pep_ids)




# Plot Pseudo ROC of the df including entries with q in x = [0,xMax] and return area under the curve
def pseudoROC(df, xMax = 0.05, onlyFirstRank = True, onlyVals = False, qColName = 'q-val', rankColName = 'Rank', title = '', label = '', plot = True):
    if (onlyFirstRank):
        qVals = df.loc[(df[qColName] <= xMax) & (df[rankColName] == 1), qColName].values.tolist()
    else: 
        qVals = df[df[qColName] <= xMax][qColName].values.tolist()
    qVals.sort()
    if (onlyVals):
        return qVals
    if (not plot):
        return auc(qVals, range(len(qVals)))
    plt.xlim(0,xMax)
    plt.ylim(0,len(qVals))
    plt.title(title)
    AUC = auc(qVals, range(len(qVals)))
    if (label != ''):
        label += ', '
    plt.plot(qVals, range(len(qVals)), label = label + 'AUC: ' + str(round(AUC, 2)))
    plt.legend(loc = 'best')
    plt.show()
    return AUC
    
# plot pseudoROCs for XLs and Non XLs and return the areas under the curves
def evalXL(df, qColName = 'class-specific_q-val', plot = True, maxQ=0.05):
    if (plot):
        nXLauc = pseudoROC(df[df['NuXL:isXL'] == 0], qColName = qColName, label = 'not cross-linked', xMax=maxQ)
        XLauc = pseudoROC(df[df['NuXL:isXL'] == 1], qColName = qColName, label = 'cross-linked', xMax=maxQ)
        plt.legend(loc = 'best')  
    else:
        nXLauc = pseudoROC(df[df['NuXL:isXL'] == 0], qColName = qColName, plot = False, xMax=maxQ)
        XLauc = pseudoROC(df[df['NuXL:isXL'] == 1], qColName = qColName, plot = False, xMax=maxQ)
    return [nXLauc, XLauc]


idCol = 'SpecId'
scoreCol = 'NuXL:score'
excluded = ['SpecId', 'Label', 'ScanNr', 'Peptide', 
            'Proteins', 'FDR', 'q-val', 'class-specific_q-val', 
            'Rank', "protein_references", 
            'CountSequenceIsTop', 'CountSequenceCharges', 'CountSequenceIsXL', 'CountSequenceIsPeptide', # tend to overfit
            'NuXL:total_Morph', 'NuXL:total_HS', 'NuXL:total_MIC', # redundant
            'A_136.062309999999997', 'A_330.060330000000022',
            'C_112.051079999999999', 'C_306.04910000000001',
            'G_152.057230000000004', 'G_346.055250000000001',
            'U_113.035089999999997', 'U_307.033110000000022',
            'NuXLScore_score',
            'NuXL:z1 mass', 'NuXL:z2 mass', 'NuXL:z3 mass', 'NuXL:z4 mass',
            "NuXL:NA", "NuXL:NT", "NuXL:localization_scores", "NuXL:best_localization", 
            'NuXL:best_localization_score', "CalcMass", "NuXL:Da difference",
             'NuXL:XL_U', 'NuXL:XL_C', 'NuXL:XL_G','NuXL:XL_A'
            ]

input_file = sys.argv[1]
raw_data = None
if input_file.endswith("idXML"):
    raw_data = readAndProcessIdXML(input_file, idCol, scoreCol, excludedCols = excluded)
else:
    raw_data = readAndProcessTSV(input_file, idCol, scoreCol, excludedCols = excluded)

print(raw_data.head())

# In[2]:
# drop lower ranks
raw_data = raw_data[raw_data["Rank"]==1] # remove all except rank 1
raw_data.reset_index(drop=True, inplace=True)

# scoring columns that can be used for scoring
print("Scoring columns:")
scoring_columns = [x for x in list(raw_data.columns) if (x not in excluded)]
print(scoring_columns)

X = raw_data[scoring_columns]
y = raw_data.Label

assert X["ExpMass"].equals(raw_data["ExpMass"]), "Order of rows changed." # important for reannotation - X must have exactly same order

# set zeros in XL columns of non-cross-links to nan
xl_features = [x for x in X.columns if "NuXL:pl_" in x or "tag_" in x or "NuXL:marker_ions_score" in x or "NuXL:partial_loss_score" in x]
X.loc[X["NuXL:isXL"]==0, xl_features] = np.nan

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)


# In[3]:
############################################################
# estimate sample weights

data = pd.concat([X, y], axis=1)
all_nXL_t=sum((data["NuXL:isXL"]==1.0) & (data["Label"] == 1.0))
all_nXL_d=sum((data["NuXL:isXL"]==1.0) & (data["Label"] == 0.0))

all_nPep_t=sum((data["NuXL:isXL"]==0.0) & (data["Label"] == 1.0))
all_nPep_d=sum((data["NuXL:isXL"]==0.0) & (data["Label"] == 0.0))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

#print(X_train.head())
#print(y_train.head())

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

print(train.columns)

train_nXL_t=sum((train["NuXL:isXL"]==1.0) & (train["Label"] == 1.0))
train_nXL_d=sum((train["NuXL:isXL"]==1.0) & (train["Label"] == 0.0))

train_nPep_t=sum((train["NuXL:isXL"]==0.0) & (train["Label"] == 1.0))
train_nPep_d=sum((train["NuXL:isXL"]==0.0) & (train["Label"] == 0.0))

#print(train_nXL_t)
#print(train_nXL_d)
#print(train_nPep_t)
#print(train_nPep_d)

diff_XL = train_nXL_t - train_nXL_d 
diff_Pep = train_nPep_t - train_nPep_d
# prevent negative values and too little influnce of targets
diff_XL = max(100, diff_XL) 
diff_Pep = max(100, diff_Pep)

wXL_t = diff_XL / train_nXL_d
wPep_t = diff_Pep / train_nPep_d

def getWeights(Xy, wPep, wXL):
    #print(Xy.columns)
    weights = list()
    for index, row in Xy.iterrows():
        #print(row)
        if row.loc["Label"] < 0.5: # decoy
            weights.append(1.0)
            continue       
        if row.loc["NuXL:isXL"] > 0.5:
            weights.append(wXL)
            continue
        else:
            weights.append(wPep)
            continue
    return weights

train_sample_weights = getWeights(train, wPep_t, wXL_t)
test_sample_weights = getWeights(test, wPep_t, wXL_t)

diff_XL = all_nXL_t - all_nXL_d 
diff_Pep = all_nPep_t - all_nPep_d
# prevent negative values and too little influnce of targets
diff_XL = max(100, diff_XL) 
diff_Pep = max(100, diff_Pep)

wXL_t = diff_XL / all_nXL_d
wPep_t = diff_Pep / all_nPep_d

all_sample_weights = getWeights(data, wPep_t, wXL_t)

print("Peptide targets: " + str(all_nPep_t))
print("Peptide decoys: " +  str(all_nPep_d))
print("XL targets: " + str(all_nXL_t))
print("XL decoys: " +  str(all_nXL_d))

print("Weight peptides: " + str(wPep_t))
print("Weight XLs: " +  str(wXL_t))

#print(train_sample_weights)
assert len(train_sample_weights) == len(X_train), "sample weights should match training data size'"


# In[7]:


############################################################
# build histogram gradient boosted classifier with monotonicity constraints

monotonic_cst= [0] * len(X.columns)

positive = ["NuXL:score", "NuXL:mass_error_p", "NuXL:total_loss_score", "NuXL:partial_loss_score", "NuXL:modds", "NuXL:precursor_score", 
            "NuXL:MIC", "NuXL:total_MIC", "NuXL:ladder_score", "NuXL:sequence_score", "NuXL:total_Morph",
            "marker_ions_score", "NuXL:pl_modds", "NuXL:pl_MIC", "NuXL:pl_im_MIC",
            #precursor intensity doesn't seem to be predictive
            "precursor_purity"
            ]

negative = ["nr_candidates", "variable_modifications", "NuXL:wTop50",
            "peplen", "isotope_error",
            "absdm", 'NuXL:err', 'NuXL:pl_err', "OMS:precursor_mz_error_ppm"
            ]

for p in positive:
    if p in X.columns.tolist():
        monotonic_cst[X.columns.tolist().index(p)] = 1

for n in negative:
    if n in X.columns.tolist():
        monotonic_cst[X.columns.tolist().index(n)] = -1

assert len(X_train) == len(y_train), "Row dimension mismatch"
assert len(X_train) == len(train_sample_weights), "Row dimension mismatch"

#clf = HistGradientBoostingClassifier(early_stopping=True, scoring="loss", monotonic_cst=monotonic_cst)
# for comparison fit without sample weights
#clf.fit(X_train, y_train)
#print("Accuracy on train data: {:.2f}".format(clf.score(X_train, y_train)))
#print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))
#print(confusion_matrix(y_test, clf.predict(X_test)))

# now with sample weights
#clf.fit(X_train, y_train, sample_weight=train_sample_weights)
#print("Accuracy on train data (sample weight): {:.2f}".format(clf.score(X_train, y_train, sample_weight=train_sample_weights)))
#print("Accuracy on test data (sample weight): {:.2f}".format(clf.score(X_test, y_test, sample_weight=test_sample_weights)))
#print(confusion_matrix(y_test, clf.predict(X_test)))

best_nXLauc = 0.0
best_XLauc = 0.0
best_alpha = 1.0
best_beta = 1.0
best_clf = None

for alpha in np.arange(0.2, 1.01, 0.2):
    for beta in np.arange(0.2, 1.01, 0.2):    
        all_sample_weights = getWeights(data, alpha*wPep_t, beta*wXL_t)

        clf = HistGradientBoostingClassifier(scoring="f1", monotonic_cst=monotonic_cst, tol = 1e-7, random_state=42, validation_fraction=None)#, early_stopping=True)
        clf.fit(X, y, sample_weight=all_sample_weights)
        print("alpha,beta: " + str(alpha) + "\t" + str(beta))
        print("Loss on all data (sample weight): {:.2f}".format(clf.score(X, y, sample_weight=all_sample_weights)))

        p = clf.predict_proba(data.loc[:, data.columns != 'Label'])[:,1] # prob for class=1 (target)
        p = pd.DataFrame({'p-value': p })
        data.reset_index(drop=True, inplace=True)
        p.reset_index(drop=True, inplace=True)
        data2 = pd.concat([data, p], axis=1)
        data2 = calcQ(data2, scoreColName="p-value")
        data2["Rank"] = 1
        # store best fit
        nXLauc, XLauc = evalXL(data2, plot=False, maxQ=0.1)
        print("pAUC(peptides), pAUC(XLs): " + str(nXLauc) + "\t" + str(XLauc))
        print("sum(pAUC): " + str(nXLauc + XLauc))
        print("Confusion matrix:")
        print(confusion_matrix(y, clf.predict(X)))

        if nXLauc + 10.0*XLauc > best_nXLauc + 10.0*best_XLauc: # we weight XL auc higher than peptide auc
            best_nXLauc = nXLauc
            best_XLauc = XLauc
            best_alpha = alpha
            best_beta = beta
            best_clf = deepcopy(clf)
        
print("Best alpha, beta: " + str(alpha) + "\t" + str(beta))
print("pAUC(peptides), pAUC(XLs): " + str(best_nXLauc) + "\t" + str(best_XLauc))
print("sum(pAUC): " + str(best_nXLauc+best_XLauc))

p = best_clf.predict_proba(data.loc[:, data.columns != 'Label'])[:,1] # prob for class=1 (target)
p = pd.DataFrame({'p-value': p})
data.reset_index(drop=True, inplace=True)
p.reset_index(drop=True, inplace=True)

assert raw_data["ExpMass"].equals(data["ExpMass"]), "Order of rows changed." # important for reannotation - must have exactly same order

#plot best fit
tmp = pd.concat([data, p], axis=1)
tmp = calcQ(tmp, scoreColName="p-value")
tmp["Rank"] = 1
nXLauc, XLauc = evalXL(tmp, plot=True)

result = pd.concat([raw_data, p], axis=1)
result = calcQ(result, scoreColName="p-value")
print(result.head())
output_file = input_file + "_nuxl.idXML"
storeIdXML(input_file, result, output_file)

r = result.loc[(result["NuXL:isXL"]==1) & (result["Label"]==1), ] 
r.to_csv(input_file + '_nuxl.csv') 

exit()

# evaluate feature importance
from sklearn.inspection import permutation_importance
result = permutation_importance(best_clf, data.loc[:, data.columns != 'Label'], data.loc[:, data.columns == 'Label'], n_repeats=10, random_state=42)
perm_sorted_idx = result.importances_mean.argsort()
fig, ax1 = plt.subplots()
ax1.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=X.columns[perm_sorted_idx])
fig.savefig(input_file + '_importance_nuxl.png', bbox_inches='tight')

plt.close(fig)    # close the figure window


exit()

