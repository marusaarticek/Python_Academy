import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU' , 'STEVEC', 'MIN_VREDNOST']

df = pd.read_csv('Naloga.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

from datetime import datetime

df['DATUM'] = pd.to_datetime(df['DATUM'])
  
test=df[df['DATUM'].dt.month == 12] #train
train=df[df['DATUM'].dt.month != 12]

x_train = train[['PRA_DAN', 'PRA_PRED', 'BDP1', 'BDP3', 'URA_VEC', 'MIN_PRET_TED', 'TEMP_MAX', 'MAX_VRED_LM1', 'PMIN_VRED_LM1', 'PMAX_VRED_LM1']]
x_test=test[['PRA_DAN', 'PRA_PRED', 'BDP1', 'BDP3', 'URA_VEC', 'MIN_PRET_TED', 'TEMP_MAX', 'MAX_VRED_LM1', 'PMIN_VRED_LM1', 'PMAX_VRED_LM1']]
y_train=train['MAX_VREDNOST']
y_test=test['MAX_VREDNOST'] 


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

#c = lr.intercept_
#m = lr.coef_
#print(c, m)

y_pred_train = lr.predict(x_train)
 
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

meanAbErr = metrics.mean_absolute_error(y_train,y_pred_train)
meanPerErr = metrics.mean_absolute_percentage_error(y_train, y_pred_train)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
smape_ = smape(y_train,y_pred_train)

print("Napake in sample:")

print('SMAPE:', smape(y_train,y_pred_train))
print('MAE: Mean Absolute Error:', meanAbErr)
print('RMSE: Root Mean Square Error:', rootMeanSqErr)
print('MAPE:', meanPerErr)


#-------------test

lr = LinearRegression()
lr.fit(x_test, y_test)
  
y_pred_test = lr.predict(x_test)
  
meanAbErr2 = metrics.mean_absolute_error(y_test,y_pred_test)
meanPerErr2 = metrics.mean_absolute_percentage_error(y_test, y_pred_test)
rootMeanSqErr2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
smape_2 = smape(y_test,y_pred_test)
 
#-----------

print("\nNapake test podatki:")

print('SMAPE:', smape(y_test,y_pred_test))
print('MAE: Mean Absolute Error:', meanAbErr)
print('RMSE: Root Mean Square Error:', rootMeanSqErr)
print('MAPE:', meanPerErr)

dff = pd.DataFrame(
    {   

        'mae': [meanAbErr],
        'mape': [meanPerErr],
        'rmse': [rootMeanSqErr],
        'smape': [smape_],
        'mae2': [meanAbErr2],
        'mape2': [meanPerErr2],
        'rmse2': [rootMeanSqErr2],
        'smape2': [smape_2]
    }
)

dff.to_csv("NapakeUcnoTestno.csv",decimal="." )

df_r = pd.read_csv('NapakeUcnoTestno.csv',   header=0 )
print(df_r.head())

#-----------SVM

#need regression model instead of classification model
# SVM --> SVR

from seaborn import load_dataset, pairplot
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

clf = SVR(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

meanAbErr = metrics.mean_absolute_error(y_test,y_pred)
meanPerErr = metrics.mean_absolute_percentage_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
smape_ = smape(y_test,y_pred)

print("\nNapake SVM:")

print('SMAPE:', smape(y_test,y_pred))
print('MAE: Mean Absolute Error:', meanAbErr)
print('RMSE: Root Mean Square Error:', rootMeanSqErr)
print('MAPE:', meanPerErr)


dff = pd.DataFrame(
    {   
        'mae': [meanAbErr],
        'mape': [meanPerErr],
        'rmse': [rootMeanSqErr],
        'smape': [smape_]
    }
)

dff.to_csv("SVM_napake.csv")

df_r = pd.read_csv('SVM_napake.csv',   header=0 )
print(df_r.head())