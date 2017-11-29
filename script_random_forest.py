# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:07:19 2017

Kaggle - House Price Prediction: Advanced Regression Techniques

@author: Ade Kurniawan
"""

import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    y = train['SalePrice']
    y.index = train['Id']
    X = train.drop('SalePrice', axis = 1)
    X.set_index('Id', inplace = True)
    test.set_index('Id', inplace = True)
    features = list(X.columns)                  # extract all features
    quantitative = list(X.describe().columns)    # obtain quantitative features
    category = [val for val in features if val not in quantitative] # obtain categorical features
    return X, y, test, quantitative, category

def clean_data(X):
    features = list(X.columns)                  # extract all features
    quantitative = list(X.describe().columns)    # obtain quantitative features
    category = [val for val in features if val not in quantitative] # obtain categorical features
    for var in features:
        if X[var].isnull().any():
            if var in quantitative:
                median = X[var].median()
                mean = X[var].mean()
                std = X[var].std()
                if abs(median-mean)<std:
                    X[var].fillna(median, inplace = True)
                else:
                    X[var].fillna(mean, inplace = True)
            elif var in category:
                if var == 'MasVnrType':
                    X[var].fillna(X[var].mode()[0], inplace = True)
                else:
                    X[var].fillna('NA', inplace = True)
    return X

def create_poly(X, deg_dict):
    '''
    a function to create polynomials from selected features in the dataset
    deg_dict should contain column-degree pairs, for example:
        deg_dict = {'A':2, 'B':3, 'C':2}
    The dataset inputted into this function must be cleaned first
    '''
    from sklearn.preprocessing import PolynomialFeatures
    for j in deg_dict:
        pf = PolynomialFeatures(degree=deg_dict[j])
        pf.fit(X[j].values.reshape(-1,1))
        cols = [j+'pow'+str(i) for i in range(2,deg_dict[j]+1)]
        X[cols] = pd.DataFrame(pf.transform(X[j].values.reshape(-1,1)), index = X.index).iloc[:,2:]
    return X

def scale_data(X_train, X_test):
    train_id = X_train.index
    test_id = X_test.index
    features = list(X_train.columns)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), 
                           index = train_id, 
                           columns = features)
    X_test = pd.DataFrame(scaler.transform(X_test), 
                           index = test_id, 
                           columns = features)
    return X_train, X_test

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    return (((np.log(y+1)-np.log(y_pred+1))**2).sum()/len(y))**0.5

#selected quantitative variables
#['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', '1stFlrSFpow2', '1stFlrSFpow3', '2ndFlrSFpow2', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'FullBath', 'FullBathpow2', 'GarageArea', 'GarageAreapow2', 'GarageCars', 'GarageCarspow2', 'GarageYrBlt', 'GrLivArea', 'GrLivAreapow2', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'MasVnrArea', 'LowQualFinSF', 'MSSubClass', 'MiscVal', 'MoSold', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'OverallQualpow2', 'ScreenPorch', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold', 'TotalBsmtSFpow2', 'TotalBsmtSFpow3', 'YearBuiltpow2']
    

X, y, test, quant, cat = load_data()
# I only use quantitative data for training now (note: some quantitative data
# here actually categorical, but the categories are stated using numbers)
X = clean_data(X)[quant]
test = clean_data(test)[quant]
deg = {'1stFlrSF':3,'2ndFlrSF':2, 
       'FullBath':2, 'GarageArea':2,
       'GarageCars':2, 'GrLivArea':2,
       'OverallQual':2, 'TotalBsmtSF':3,
       'YearBuilt':2}
X = create_poly(X, deg)
test = create_poly(test, deg)
#X, test = scale_data(X, test)
selected_features = ['1stFlrSFpow2',
                     '1stFlrSFpow3',
                     '2ndFlrSFpow2',
                     'BedroomAbvGr',
                     'BsmtFullBath',
                     'BsmtHalfBath',
                     'Fireplaces',
                     'FullBath',
                     'FullBathpow2',
                     'GarageAreapow2',
                     'GarageCarspow2',
                     'GarageYrBlt',
                     'GrLivAreapow2',
                     'HalfBath',
                     'KitchenAbvGr',
                     'LotArea',
                     'LotFrontage',
                     'MSSubClass',
                     'MasVnrArea',
                     'MoSold',
                     'OpenPorchSF',
                     'OverallCond',
                     'OverallQual',
                     'OverallQualpow2',
                     'ScreenPorch',
                     'TotalBsmtSFpow2',
                     'TotalBsmtSFpow3',
                     'WoodDeckSF',
                     'YearBuilt',
                     'YearBuiltpow2',
                     'YearRemodAdd',
                     'YrSold']
X = X[selected_features]
test = test[selected_features]
# the training part
# importing necessary modules, I want to use Ridge and Lasso and then compare which
# one gives the best result
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
params = {'n_estimators':[10, 50, 100],
          'max_features': ['auto','log2',None],
          'bootstrap': [True, False]}
gsRF = GridSearchCV(estimator = rf, param_grid = params)

#training the model
gsRF.fit(X,y)

#prediction and result
pred = gsRF.predict(X)
err = rmsle(y, pred)    #calculating root mean square logarithmic error
print('Result:')
print('Best parameter: ',gsRF.best_params_)
print('Best score: ',gsRF.best_score_)
print('Root mean square logarithmic error: ', err)
print('\n')

# generate the submission file
#test_rf = pd.DataFrame(gsRF.predict(test), index = test.index, columns = ['SalePrice'])
#test_rf.to_csv('rf_unnormalized.csv')