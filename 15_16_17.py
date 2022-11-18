import csv
import json
import time

def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []
    with open(csvFilePath, encoding='utf-8') as csvf: 
        csvReader = csv.DictReader(csvf) 

        for row in csvReader: 
            jsonArray.append(row)
  
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)
          
csvFilePath = r'podatki_python_akademija.csv'
jsonFilePath = r'data.json'

start = time.perf_counter()
csv_to_json(csvFilePath, jsonFilePath)

import pandas as pd

with open('data.json') as data:
    df=pd.read_json(data,orient='columns')
    df.to_csv('not_one_line.csv')
    print(df.head())

