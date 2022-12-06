import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import csv 



columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

df = pd.read_csv('podatki_python_akademija.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

x = df.drop(columns = 'MAX_VREDNOST')
y=df['MAX_VREDNOST']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 100)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae


model = make_pipeline(
    StandardScaler(), LassoLarsIC(criterion="bic", normalize=False)
).fit(x, y)

pred = model.predict(x)
 
c = []
c.append(model[-1].coef_)
fields =df.drop(columns = 'MAX_VREDNOST')


filename = "Rezultat44.csv"
rows = c
newdf=pd.DataFrame()

for i in fields:
    newdf[i] = 0
#print(newdf)

# writing to csv file 
with open(filename, 'w', newline='') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)

df = pd.read_csv('Rezultat44.csv', delimiter=";", decimal=",", header=0)
print(df.head())


#-------!!!!!!!!!!!!!!!
#it picks the values itself, if the value is redundant its coeficient is set to 0.



from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae


mae_ = mae(y, pred)

def mape(b, pred): 
    b, pred = np.array(b), np.array(pred)
    return np.mean(np.abs((b - pred) / b)) * 100
mape_ =mape(y,pred)

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)
smape_ =smape(y, pred)

def rmse(b, pred): 
    b, pred = np.array(b), np.array(pred)
    return np.sqrt(np.square(np.subtract(b,pred)).mean())
rmse_=rmse(y,pred)
 
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

df_r = pd.read_csv('Rezultat44.csv', delimiter=";", decimal=",", header=0 )
df_r['MAE'], df_r['MAPE'],df_r['RMSE'],df_r['SMAPE'] =  [mae_],[mape_],[rmse_],[smape_]
df_r.to_csv('Rezultat44.csv', sep=',')