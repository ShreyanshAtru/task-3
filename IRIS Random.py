# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:43:45 2021

@author: HP
"""
import pandas
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir('D:\hose')
df = pd.read_csv('IRIS.csv')
df.head()

###test train split 

cols = df.columns
x_cols = cols[:4]
y_cols = cols[4]

X = df[x_cols]
y = df[y_cols]

train = 0.8
test = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train, test_size=test)


##fitting the algorithm
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
score = model.score(X_test, y_test)
percentage = score * 100
print("%.3f" % percentage, "%")


##confusion matrix 
cm = confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')