import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

df = pd.read_csv('podatki_python_akademija.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

x = df.drop(columns = 'MAX_VREDNOST')
y=df['MAX_VREDNOST']


# izberemo atribute glede na vrednost p? (parameter ki pove..?)
# ce je p > 0.05 removas atribut iz lista 
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html

selected_features = list(x.columns)
pmax = 1
while (len(selected_features)>0):
    p= []
    x_1 = x[selected_features]
    x_1 = sm.add_constant(x_1)
    model = sm.OLS(y,x_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = selected_features)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        selected_features.remove(feature_with_p_max)
    else:
        break  
        
#print(selected_features)
#['PRA_DAN', 'PRA_PRED', 'BDP1', 'BDP3', 'URA_VEC', 'MIN_PRET_TED', 'TEMP_MAX', 'MAX_VRED_LM1', 'PMIN_VRED_LM1', 'PMAX_VRED_LM1']

#Model z najbolsimi vrednostmi
a=df[selected_features]
b=df['MAX_VREDNOST']

regr = linear_model.LinearRegression()
regr.fit(a, b)
#shranimo koeficiente spremenljivk
c = []
c.append(regr.coef_)

#--------------v datoteko
import csv 


fields = [selected_features]
rows = c

filename = "Rezultat.csv"
#zapis v csv dat
with open(filename, 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(fields) 
    csvwriter.writerows(rows)


#-----------napake

from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae

lm = linear_model.LinearRegression()
lm.fit(a, b)
pred = lm.predict(a)


mae_ = mae(b, pred)

def mape(b, pred): 
    b, pred = np.array(b), np.array(pred)
    return np.mean(np.abs((b - pred) / b)) * 100
mape_ =mape(b,pred)

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)
smape_ =smape(b, pred)

def rmse(b, pred): 
    b, pred = np.array(b), np.array(pred)
    return np.sqrt(np.square(np.subtract(b,pred)).mean())
rmse_=rmse(b,pred)
 
def vector(values, pred): 
    mae1 = mae(values, pred)
    mape1 =mape(values, pred)
    smape1 =smape(values, pred)
    rmse1 =rmse(values, pred)
    list1 = [mae1,mape1,smape1,rmse1]
    vektor = np.array(list1)
    return vektor
 
data = {'mae': [mae_],
        'mape': [mape_],
        'rmse': [rmse_],
        'smape': [smape_]}

df_r = pd.read_csv('Rezultat.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)
df_r['MAE'], df_r['MAPE'],df_r['RMSE'],df_r['SMAPE'] =  [mae_],[mape_],[rmse_],[smape_]
df_r.to_csv('Rezultat.csv', sep=',')

print("Ne vem kako prvilno zdruzit oba dataframa v eno csv datoteko:/")