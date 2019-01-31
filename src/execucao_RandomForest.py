# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:04:29 2019

@author: lfreire
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

dataset_treino = pd.read_csv('../data/train_clean.csv')
dataset2_test = pd.read_csv('../data/test.csv')

# Campos utilizados - quartos,suites, vagas, area_util, piscina, ginastica, vista_mar

X_train = dataset_treino.iloc[:, [4,5,6,7,12,18,17,19]].values
Y_train = dataset_treino.iloc[:,-1].values

X_test = dataset2_test.iloc[:, [4,5,6,7,12,18,17,19]].values


regr = RandomForestRegressor(max_depth=7, random_state=6,n_estimators=40)
regr.fit(X_train, Y_train)
pred_train = regr.predict(X_train)
pred_result = regr.predict(X_test).astype(int)

pd.DataFrame(pred_result, columns=['preco']).to_csv('prediction_rf_7640.csv')

print('Arquivo gerado com sucesso!')

