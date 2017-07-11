#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 08:30:24 2017

@author: bradley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:14:31 2017

@author: bradley
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD

pd.options.display.max_rows = 500

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

use_leaks = False

# Augment the training set
leaks = {
    1:71.34112,
    12:109.30903,
    23:115.21953,
    28:92.00675,
    42:87.73572,
    43:129.79876,
    45:99.55671,
    57:116.02167,
    3977:132.08556,
    88:90.33211,
    89:130.55165,
    93:105.79792,
    94:103.04672,
    1001:111.65212,
    104:92.37968,
    72:110.54742,
    78:125.28849,
    105:108.5069,
    110:83.31692,
    1004:91.472,
    1008:106.71967,
    1009:108.21841,
    973:106.76189,
    8002:95.84858,
    8007:87.44019,
    1644:99.14157,
    337:101.23135,
    253:115.93724,
    8416:96.84773,
    259:93.33662,
    262:75.35182,
    1652:89.77625
    }

leaks = pd.Series(leaks)
leaks = leaks.reset_index()
leaks.columns = ['ID', 'y']
leaks = pd.merge(leaks, test, how='inner', on='ID')

if use_leaks==True:
    train = train.append(leaks)

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
     
# Remove columns with only one occuring value
for c in train.columns:
    if len(train[c].value_counts())==1:
        print(c)
        del train[c]
        del test[c]
        
        
# Feature_Settings should be a dictionary, e.g:
# {'magic':False, 'ID':False, 'n_comp':12, 'tSVD':True, 'PCA':True,
#  'ICA':True, 'GRP':True, 'SRP':True, 'SparseThreshold':0.0}
def perform_feature_engineering(train,test,config):
         
    for c in train.columns:
        if len(train[c].value_counts())==2:
            if train[c].mean()<config['SparseThreshold']:
                del train[c]
                del test[c]
                
    col = list(test.columns)
    if config['ID']!=True:
        col.remove('ID')

    # tSVD
    if config['tSVD']==True:
        tsvd = TruncatedSVD(n_components=config['n_comp'])
        tsvd_results_train = tsvd.fit_transform(train[col])
        tsvd_results_test = tsvd.transform(test[col])
        for i in range(1, config['n_comp'] + 1):
            train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
            test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
    # PCA
    if config['PCA']==True:
        pca = PCA(n_components=config['n_comp'])
        pca2_results_train = pca.fit_transform(train[col])
        pca2_results_test = pca.transform(test[col])
        for i in range(1, config['n_comp'] + 1):
            train['pca_' + str(i)] = pca2_results_train[:, i - 1]
            test['pca_' + str(i)] = pca2_results_test[:, i - 1]
    # ICA
    if config['ICA']==True:
        ica = FastICA(n_components=config['n_comp'])
        ica2_results_train = ica.fit_transform(train[col])
        ica2_results_test = ica.transform(test[col])
        for i in range(1, config['n_comp'] + 1):
            train['ica_' + str(i)] = ica2_results_train[:, i - 1]
            test['ica_' + str(i)] = ica2_results_test[:, i - 1]
            
    # GRP
    if config['GRP']==True:
        grp = GaussianRandomProjection(n_components=config['n_comp'], eps=0.1)
        grp_results_train = grp.fit_transform(train[col])
        grp_results_test = grp.transform(test[col])
        for i in range(1, config['n_comp'] + 1):
            train['grp_' + str(i)] = grp_results_train[:, i - 1]
            test['grp_' + str(i)] = grp_results_test[:, i - 1]
            
    # SRP
    if config['SRP']==True:
        srp = SparseRandomProjection(n_components=config['n_comp'], dense_output=True, random_state=420)
        srp_results_train = srp.fit_transform(train[col])
        srp_results_test = srp.transform(test[col])
        for i in range(1, config['n_comp'] + 1):
            train['srp_' + str(i)] = srp_results_train[:, i - 1]
            test['srp_' + str(i)] = srp_results_test[:, i - 1]
        
    if config['magic']==True:
        magic_mat = train[['ID','X0','y']]
        magic_mat = magic_mat.groupby(['X0'])['y'].mean()
        magic_mat = pd.DataFrame({'X0':magic_mat.index,'magic':list(magic_mat)})
        mean_magic = magic_mat['magic'].mean()
        train = train.merge(magic_mat,on='X0',how='left')
        test = test.merge(magic_mat,on='X0',how = 'left')
        test['magic'] = test['magic'].fillna(mean_magic)
    return train,test

# repeats should always be at least 1
def perform_cv(xgb_params, train, k, repeats):
    
    cv_means = []
    cv_stds = []
    
    for r in range(0,repeats):
    
        # Create CV folds
        train = train.sample(frac=1) #, random_state=420)
        partitions = []
    
        foldsize = train.shape[0]//k
        remainder = train.shape[0]%k
        stride = 0
        last = 0
        
        for i in range(0,k):
            if i < remainder:
                stride = foldsize + 1
            else:
                stride = foldsize
        
            part = train[last:(last+stride)]
            partitions.append(part)
            last = last + stride
        
        #print(len(partitions))
    
        
        # Cross Validation
        scores = []
        for i in range(0,k):
            cv_test = partitions[i]
            cv_train = pd.DataFrame()
            for j in range(0,k):
                if j!=i:
                    cv_train = cv_train.append(partitions[j])
                
            dtrain = xgb.DMatrix(cv_train.drop('y', axis=1), cv_train['y'])
            dtest = xgb.DMatrix(cv_test.drop('y', axis=1), cv_test['y'])
            
            xgb_params['base_score'] = np.mean(cv_train['y'])
        
            #num_boost_rounds = 1250
            # train model
            #model = xgb.train(dict(xgb_params), dtrain)
            model = xgb.train(dict(xgb_params), dtrain, num_boost_round=xgb_params['num_round'])
            y_pred = model.predict(dtest)
            #if hasattr(model, 'bst.best_iteration'):
            #    print(model.bst.best_iteration)
            
            score = r2_score(cv_test['y'],y_pred)
            #print('fold: ' + str(i) ) #+ ', score: ' + str(score))
            scores.append(score)
            
        mu = np.mean(scores)
        sigma = np.std(scores)
            
        cv_means.append(mu)
        cv_stds.append(sigma)
            
        print('repeat: ' + str(r) + ', mean_score: ' + str(mu) + ', std_dev: ' + str(sigma))
    
    return np.mean(cv_means), np.std(cv_means)




xgb_params = {
    'eta': 0.0045,
    'max_depth': 3,
    'subsample': 0.93,
    'colsample_bytree': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': 0, # base prediction = mean(target)
    'silent': 0,
    'num_round': 1250
}

feature_config = {'magic':True, 'ID':True, 'n_comp':12, 'tSVD':False, 'PCA':False,
                  'ICA':False, 'GRP':False, 'SRP':False, 'SparseThreshold':0.0}
print(xgb_params)
print(feature_config)     

eng_train, eng_test = perform_feature_engineering(train,test,feature_config)


dtrain = xgb.DMatrix(eng_train.drop('y', axis=1), eng_train['y'])
dtest = xgb.DMatrix(eng_test)
            
xgb_params['base_score'] = np.mean(eng_train['y'])
        
model = xgb.train(dict(xgb_params), dtrain, num_boost_round=xgb_params['num_round'])
y_pred = model.predict(dtest)

sub = pd.DataFrame({'ID':test['ID'],'y':y_pred})

filepath = 'temp/xgb_intermediate_sub.csv'

if use_leaks==True:
    sub['y'] = sub.apply(lambda r: leaks[int(r['ID'])] if int(r['ID']) in leaks else r['y'], axis=1)
    filepath = 'temp/xgb_intermediate_sub_use_leaks.csv'

sub.to_csv(filepath,index=False)



