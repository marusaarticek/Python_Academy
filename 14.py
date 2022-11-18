import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from datetime import datetime
from datetime import timedelta


columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU' , 'STEVEC', 'MIN_VREDNOST']
df = pd.read_csv('podatki_python_akademija.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)


df['DATUM'] = pd.to_datetime(df['DATUM'])
df['DATUM'] = pd.to_datetime(df['DATUM'], format='%Y-%m-%d')
  
                   
def linreg(x_train, y_train, x_test, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    y_pred_train = lr.predict(x_test)

    from sklearn import metrics
    from sklearn.metrics import mean_absolute_percentage_error

    meanPerErr = metrics.mean_absolute_percentage_error(y_test, y_pred_train)

    print('MAPE:', meanPerErr)


date1=pd.to_datetime('2012-02-01', format='%Y-%m-%d')
date2=pd.to_datetime('2012-11-30', format='%Y-%m-%d')
date1_end=pd.to_datetime('2012-01-07', format='%Y-%m-%d')
date2_end=pd.to_datetime('2012-11-05', format='%Y-%m-%d')

while date1 >= date1_end:
    filtered_train = df.loc[(df['DATUM'] >= date1)
                     & (df['DATUM'] < date2)]
    filtered_test = df.loc[(df['DATUM'] >= '2012-12-01')
                     & (df['DATUM'] < '2012-12-31')]
    x_train = filtered_train[['PRA_DAN', 'PRA_PRED', 'BDP1', 'BDP3', 'URA_VEC', 'MIN_PRET_TED', 'TEMP_MAX', 'MAX_VRED_LM1', 'PMIN_VRED_LM1', 'PMAX_VRED_LM1']]
    x_test=filtered_test[['PRA_DAN', 'PRA_PRED', 'BDP1', 'BDP3', 'URA_VEC', 'MIN_PRET_TED', 'TEMP_MAX', 'MAX_VRED_LM1', 'PMIN_VRED_LM1', 'PMAX_VRED_LM1']]
    y_train=filtered_train['MAX_VREDNOST']
    y_test=filtered_test['MAX_VREDNOST']

    linreg(x_train, y_train, x_test, y_test)
    
    date1 = date1 +pd.DateOffset(days=-1)
    date2 = date2 +pd.DateOffset(days=-1)
