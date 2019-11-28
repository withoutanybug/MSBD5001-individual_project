# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:45:00 2019

@author: lu
"""

import numpy as np
import pandas as pd
from sklearn import svm 

# Importing the dataset
dataset = pd.read_csv('train_ori.csv')
testdata = pd.read_csv('testdata.csv')
outcome = pd.read_csv('submission.csv')
dataset = dataset.drop(['id','tags',],axis = 1, inplace=False)
testdata = testdata.drop(['id','tags',],axis = 1, inplace=False)
dataset=dataset.dropna()
#testdata=testdata.fillna(-1,-1,-1)

genres = dataset["genres"].str.get_dummies(",")
categories = dataset['categories'].str.get_dummies(",")
dataset =pd.concat([dataset, genres,categories], axis=1)
dataset = dataset.drop(['genres','categories'],axis = 1, inplace=False)
genres1 = testdata["genres"].str.get_dummies(",")
categories1 = testdata['categories'].str.get_dummies(",")
testdata =pd.concat([testdata, genres1,categories1], axis=1)
testdata = testdata.drop(['genres','categories'],axis = 1, inplace=False)

for col2 in testdata.columns[6:]:
    if col2 not in dataset.columns[7:]:   
        testdata = testdata.drop(col2,axis = 1, inplace=False)

for col in dataset.columns[7:]:               
     if col not in testdata.columns[6:]:
         dataset = dataset.drop(col,axis = 1, inplace=False)
            
X = dataset.iloc[:, 1:].values
X_test= testdata.iloc[:, 0:].values
y = dataset.iloc[:, 0].values
X = np.insert(X, 3, values=0, axis=1)###month
X_test = np.insert(X_test, 3, values=0, axis=1)###month

(line,col)=X.shape
(line1,col1)=X_test.shape
# Splitting the dataset into the Training set and Test set

for i in range(line):
    (X[i,2],X[i,3])=X[i,2].split('/')[0],X[i,2].split('/')[1]
    X[i,2]=int(X[i,2])
    X[i,3]=int(X[i,3])
    X[i,4]=int(X[i,4].split(',')[2])

for t in range(line1):
    (X_test[t,2],X_test[t,3])=X_test[t,2].split('/')[0],X_test[t,2].split('/')[1]
    X_test[t,2]=int(X_test[t,2])
    X_test[t,3]=int(X_test[t,3])
    X_test[t,4]=int(X_test[t,4].split('/')[2])

X_train=X
y_train=y
X_val=X_test
a = np.arange(col)
X = np.delete(X,a[10:25], 1)
X_test = np.delete(X_test,a[10:25], 1)
    #Feature Scaling
from sklearn.preprocessing import MinMaxScaler
X_scaler=MinMaxScaler()
X_train = X_scaler.fit_transform(X_train)
X_val = X_scaler.transform(X_val)
y_df = []

from sklearn.ensemble import RandomForestRegressor
for i in range(100):
    rf = RandomForestRegressor(max_depth=2,n_estimators=500)
    rf.fit(X_train, y_train)
    y_pred=rf.predict(X_val)
    y_df.append(y_pred)


# =============================================================================
# rbfSVM = svm.SVR(kernel = 'rbf')
# rbfSVM.fit(X_train, y_train)
# y_pred = rbfSVM.predict(X_val)
# =============================================================================
    
np.savetxt("y.csv", y_pred, delimiter=",",header='playtime_forever',comments='')
submission = pd.read_csv('y.csv')
submission.to_csv('test1.csv',index=1,index_label='id',header=1)


###pd.to_datetime(pd.Series([X[:2],  None]))
# =============================================================================
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# labelencoder_x=LabelEncoder()
# X[:,0]=labelencoder_x.fit_transform(X[:,0])
# onehotencoder=OneHotEncoder(categorical_features=[2])
# X=onehotencoder.fit_transform(X).toarray()
# =============================================================================
# # =============================================================================
# RMSE = np.zeros((100,1))
# for i in range(100):
#     from sklearn.model_selection import train_test_split
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = i)
# 
# # Feature Scaling
#     from sklearn.preprocessing import StandardScaler
#     X_scaler=StandardScaler()
#     X_train = X_scaler.fit_transform(X_train)
#     X_val = X_scaler.transform(X_val)
# 
#     from sklearn.ensemble import RandomForestRegressor
#     rf = RandomForestRegressor(n_estimators = 10, random_state=0)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_val)
# 
#     RMSE[i]=1/line*np.linalg.norm((y_pred-y_val),ord=2)
# 
# print(np.mean(RMSE))
# =============================================================================