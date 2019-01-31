# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:04:29 2019

@author: lfreire
"""

import pandas as pd
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('../data/train_clean.csv')

# Campos utilizados - quartos,suites, vagas, area_util, piscina, ginastica, vista_mar
X = dataset.iloc[:, [4,5,6,7,12,18,17,19]].values
y = dataset.iloc[:,-1].values


X_train, X_test, Y_train, y_test = train_test_split(
        X, 
        y,
        test_size = 0.33,
        random_state = 5
)


## LinearRegression  ##########################################################
from sklearn.linear_model import LinearRegression
lm  = LinearRegression()
lm.fit(X_train,Y_train)
y_pred_train = lm.predict(X_train)
y_pred_test = lm.predict(X_test)
print('LINEAR REGRESSION')
print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(Y_train, y_pred_train) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(Y_train, y_pred_train)))
print('R2   = %.3f' %                     r2_score(Y_train, y_pred_train) )
print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , y_pred_test) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_pred_test)))
print('R2   = %.3f' %                     r2_score(y_test , y_pred_test) )
#################################################################################


## PolynomialFeatures + LinearRegression  #######################################
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(2)
X_train_poly = pf.fit_transform(X_train)
X_test_poly = pf.fit_transform(X_test)
lm  = LinearRegression()
lm.fit(X_train_poly,Y_train)
y_pred_train = lm.predict(X_train_poly)
y_pred_test = lm.predict(X_test_poly)
print('POLINOMIAL LINEAR REGRESSION')
print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(Y_train, y_pred_train) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(Y_train, y_pred_train)))
print('R2   = %.3f' %                     r2_score(Y_train, y_pred_train) )
print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , y_pred_test) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_pred_test)))
print('R2   = %.3f' %                     r2_score(y_test , y_pred_test) )
#################################################################################

## RandomForestRegressor ########################################################
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=7, random_state=6,n_estimators=40)
regr.fit(X_train, Y_train)
pred_train = regr.predict(X_train)
pred_test = regr.predict(X_test)
print('================================================')
print('RANDOM FOREST REGRESSOR')
print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(Y_train, pred_train) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(Y_train, pred_train)))
print('R2   = %.3f' %                     r2_score(Y_train, pred_train) )
print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , pred_test) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , pred_test)))
print('R2   = %.3f' %                     r2_score(y_test , pred_test) )
#################################################################################


## KNeighborsRegressor ##########################################################
from sklearn.neighbors import KNeighborsRegressor
k=4
knn = KNeighborsRegressor()
knn = KNeighborsRegressor(n_neighbors=k)
knn = knn.fit(X_train, Y_train)
y_train_pred_knn = knn.predict(X_train)
y_test_pred_knn  = knn.predict(X_test)
print('================================================')
print('KNN')
print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(Y_train, y_train_pred_knn) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(Y_train, y_train_pred_knn)))
print('R2   = %.3f' %                     r2_score(Y_train, y_train_pred_knn) )
print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , y_test_pred_knn) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_test_pred_knn)))
print('R2   = %.3f' %                     r2_score(y_test , y_test_pred_knn) )
#################################################################################


## SVM ##########################################################################
from sklearn.svm import SVR
clf = SVR(C=1, epsilon=1)
clf.fit(X_train,Y_train)
y_train_pred_svm = clf.predict(X_train)
y_test_pred_svm = clf.predict(X_test)
print('================================================')
print('SVM')
print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(Y_train, y_train_pred_svm) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(Y_train, y_train_pred_svm)))
print('R2   = %.3f' %                     r2_score(Y_train, y_train_pred_svm) )
print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , y_test_pred_svm) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_test_pred_svm)))
print('R2   = %.3f' %                     r2_score(y_test , y_test_pred_svm) )
#################################################################################

## DecisionTreeClassifier #######################################################
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=10)
clf = clf.fit(X_train, Y_train)
rid_train_pred = clf.predict(X_train)
rid_test_pred = clf.predict(X_test)
print('================================================')
print('DECISION TREE REGRESSOR')
print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(Y_train, rid_train_pred) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(Y_train, rid_train_pred)))
print('R2   = %.3f' %                     r2_score(Y_train, rid_train_pred) )

print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , rid_test_pred) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , rid_test_pred)))
print('R2   = %.3f' %                     r2_score(y_test , rid_test_pred) )
#################################################################################

## GaussianNB ###################################################################
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(X_train, Y_train)
y_train_pred_gnb = gnb.predict(X_train)
y_test_pred_gnb  = gnb.predict(X_test)
print('================================================')
print('GAUSSIAN Naive Bauyes')
print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(Y_train, y_train_pred_gnb) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(Y_train, y_train_pred_gnb)))
print('R2   = %.3f' %                     r2_score(Y_train, y_train_pred_gnb) )
print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , y_test_pred_gnb) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_test_pred_gnb)))
print('R2   = %.3f' %                     r2_score(y_test , y_test_pred_gnb) )
#################################################################################

## KMeans #######################################################################
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=10)
kmeans.fit(X_train,Y_train)
y_train_pred_kmeans = kmeans.predict(X_train)
y_test_pred_kmeans  = kmeans.predict(X_test)
print('================================================')
print('K MEANS')
print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(Y_train, y_train_pred_kmeans) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(Y_train, y_train_pred_kmeans)))
print('R2   = %.3f' %                     r2_score(Y_train, y_train_pred_kmeans) )
print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , y_test_pred_kmeans) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_test_pred_kmeans)))
print('R2   = %.3f' %                     r2_score(y_test , y_test_pred_kmeans) )
#################################################################################
