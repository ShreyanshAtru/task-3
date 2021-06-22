# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 17:07:28 2021

@author: HP
"""

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

os.chdir('D:')
df = pd.read_csv('price dataset.csv')
df.describe()
df1 = df[df['symbol']=='AMZN']
df1.head(3)
colors = ['red','blue']
sns.set(palette = colors , font = 'Arial'
        ,style = 'white' , rc = {'axes.facecolor':'whitesmoke','figure.facecolor':'whitesmoke'})
sns.palplot(colors , size = 1.5)
df['symbol'].nunique()
fig = plt.figure(figsize = (20,8))
ax = sns.lineplot(data = df1 , x = 'date' , y = 'open')
ax = sns.lineplot(data = df1 , x  = 'date' , y = 'close' , color = colors[1])
sns.pairplot(df1,corner=True)
df1.corr()
df1.corr()['close']
sns.heatmap(df1.corr(), annot=True, cmap=[colors[0],colors[1]], linecolor='white', linewidth=2 )




X=df1[['volume','open']]
y=df1['close']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=False, random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)
print(X_train , X_test)


from scipy.stats import levene, shapiro
int_cols=df1.select_dtypes(exclude='object').columns.to_list()

for i in int_cols:
    _, p_value=shapiro(df1[i])
    if p_value<0.05:
        print("Feature {} is normaly distributed".format(i))
    else:
        print("Feature {} is not normaly distributed".format(i))
        
    print("Normalitiy test p_value for featue -  {} is {}".format(i,np.round(p_value,3)))





####basic Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
lin = LinearRegression()
lin.fit(X_train , y_train)
lin_pred = lin.predict(X_test)
r2 = r2_score(y_test , lin_pred)
mse = mean_squared_error(y_test , lin_pred)
print(r2 , mse)
fig=plt.figure(figsize=(15,8))
p=pd.Series(lin_pred, index=y_test.index)
plt.plot(y_test)
plt.plot(p)
plt.legend(['y_test','lin_pred'])


#### Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(X_train , y_train)
dtr_pred = dtr.predict(X_test)
dtr_r2 = r2_score(y_test , dtr_pred)
dtr_mse = mean_squared_error(y_test,dtr_pred)
print(dtr_r2 , dtr_mse)
fig=plt.figure(figsize=(15,8))
p=pd.Series(dtr_pred, index=y_test.index)
plt.plot(y_test)
plt.plot(p)
plt.legend(['y_test','dtr_pred'])
df = pd.DataFrame({'Actual': y_test, 'Predicted': dtr_pred})
df.tail()



## lasso
from sklearn.linear_model import Lasso
l = Lasso(normalize = True)
l.fit(X_train , y_train)
l_pred = l.predict(X_test)
l_pred.mean()
score_lasso=l.score(X_test, y_test)
print(score_lasso)
df = pd.DataFrame({'Actual': y_test, 'Predicted': l_pred})
df.head()
fig=plt.figure(figsize=(15,8))
p=pd.Series(l_pred, index=y_test.index)
plt.plot(y_test)
plt.plot(p)
plt.legend(['y_test','l_pred'])

a = sns.boxplot(y_test , color = 'blue')
b = sns.boxplot(l_pred)















