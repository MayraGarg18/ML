# -*- coding: utf-8 -*-
"""Copy of Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ms09yv8Q-XSaWRTyiH7cj5tQoALsC97R
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
df = pd.read_csv('laptop_data.csv')
df.head()
df.info()
df.drop(columns = ['Unnamed: 0'], inplace = True)
df.head()
df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')
df.head()
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')
df.info()
# import seaborn as sns
# sns.displot(df['Price'])
# df['Company'].value_counts().plot(kind = 'bar')
# sns.barplot( x = df['Company'], y = df['Price'])
# plt.xticks(rotation = 'vertical')
# plt.show()
# df['TypeName'].value_counts().plot(kind = 'bar')
# sns.barplot( x = df['TypeName'], y = df['Price'])
# plt.xticks(rotation = 'vertical')
# plt.show()
# sns.displot(df['Inches'])
# sns.scatterplot(x = df['Inches'], y = df['Price'])
df['ScreenResolution'].value_counts()
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
df.head()
# df['Touchscreen'].value_counts().plot(kind = 'bar')
# sns.barplot(x = df['Touchscreen'], y = df['Price'])
df['IPS'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
df.head()
# df['IPS'].value_counts().plot(kind = 'bar')
# sns.barplot(x = df['IPS'], y = df['Price'])
new = df['ScreenResolution'].str.split('x', n=1, expand = True)
df['X_res'] = new[0]
df['Y_res'] = new[1]
df.head()
df['X_res'] = df['X_res'].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])
df.head()
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')
df.info()
# df.corr()['Price']
df['PPI'] = (((df['X_res']**2 + df['Y_res']**2)**0.5/ df['Inches'])).astype('float')
df.head()
# df.corr()['Price']
df.drop(columns= ['ScreenResolution'], inplace = True)
df.head()
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
df.head()
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text== 'Intel Core i3':
        return text
    else:
        if text.split()[0]=='Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

df['Cpu_Brand'] = df['Cpu Name'].apply(fetch_processor)
df.head()
# df['Cpu Brand'].value_counts().plot(kind = 'bar')
# sns.barplot(x = df['Cpu Brand'], y = df['Price'])
# plt.xticks(rotation = 'vertical')
# plt.show()
df.drop(columns = ['Cpu Name', 'Cpu'], inplace = True)
df.head()
# df['Ram'].value_counts().plot(kind = 'bar')
# sns.barplot(x = df['Ram'], y = df['Price'])
# plt.xticks(rotation = 'vertical')
# plt.show()
df['Memory'].value_counts()
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex = True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n=1, expand = True)

df["first"] = new[0]
df["first"] = df["first"].str.strip()



df["second"] = new[1]

df["Layer1HDD"]= df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"]= df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"]= df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"]= df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first']= df['first'].str.replace(r'\D','')
df['second'].fillna("0", inplace = True)

df["Layer2HDD"]= df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"]= df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"]= df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"]= df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second']= df['second'].str.replace(r'\D','')

df["first"]= df["first"].astype(int)
df["second"]= df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"] + df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"] + df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"] + df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"] + df["second"]*df["Layer2Flash_Storage"])

df.drop(columns = ['first', 'second', 'Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage', 'Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage'], inplace = True)
df.head()
df.drop(columns = ['Memory'], inplace = True)
df.head()
df.corr()['Price']
df.drop(columns = ['Hybrid','Flash_Storage'], inplace = True)
df.head()
df['Gpu'].value_counts()
df['Gpu brand']= df['Gpu'].apply(lambda x: x.split()[0])
df['Gpu brand'].value_counts()
df= df[df['Gpu brand'] !='ARM']
df['Gpu brand'].value_counts()
# sns.barplot(x = df['Gpu brand'], y = df['Price'])
# plt.xticks(rotation = 'vertical')
# plt.show()
df.drop(columns = ['Gpu'], inplace = True)
df.head()
df['OpSys'].value_counts()
# sns.barplot(x= df['OpSys'], y =df['Price'])
# plt.xticks(rotation ='vertical')
# plt.show()
def Categorise_OS(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS':
        return 'Mac'
    else:
        return 'Others/No OS/Linux/Android/Chrome OS'
df['os']= df['OpSys'].apply(Categorise_OS)
df.head()
df.drop(columns=['OpSys','X_res','Y_res','Inches'], inplace = True)
df.head()
# sns.barplot(x= df['os'], y =df['Price'])
# plt.xticks(rotation ='vertical')
# plt.show()
# sns.displot(df['Weight'])
# sns.scatterplot(x=df['Weight'], y=df['Price'])
df.corr()['Price']
df.corr()
df = df.rename(columns={"Cpu Brand":"Cpu_Brand",
                        "Gpu brand": "Gpu_Brand"})
# sns.heatmap(df.corr())
# sns.displot(df['Price'])
# sns.displot(np.log(df['Price']))
X = df.drop(columns=['Price'])
Y = np.log(df['Price'])
X
Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 1)
X_train
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
# pip install xgboost
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR

df.columns

from sklearn.preprocessing import StandardScaler

step1 = ColumnTransformer([
    ("num", StandardScaler(), ['Ram', 'Touchscreen', 'IPS', 'PPI', 'HDD', 'SSD']),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ['Company', 'TypeName','Weight', 'Cpu_Brand', 'Gpu_Brand', 'os']),
])
# encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
# step1 = encoder.fit_transform(X_train)
step2 = RandomForestRegressor(n_estimators = 100,
                             random_state = 3,
                             max_samples = 0.5,
                             max_features = 0.75,
                             max_depth = 15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print('r2', r2_score(Y_test, Y_pred))

df.corr()['Price']

import pickle
pickle.dump(df, open('df3.pkl','wb'))
pickle.dump(pipe, open('pipe3.pkl','wb'))