#        THE SPARK FOUNDATION INTERNSHIP

#        Author - SHREYANSH JAIN
#        perform EDA on the given dataset "sampleSuperstore"

import pandas as pd
import os
import seaborn as sns
os.chdir("E:/")
data = pd.read_csv("SampleSuperstore.csv")
data.head()
data.info()
data.isnull().sum()
#visualizing the null values

sns.heatmap(data.isnull(), yticklabels = False , cbar = False , cmap = 'viridis')

print(data['Segment'].unique())

#     there are 3 categorical values in the 'segment' column
#     converting them into inteager values

pd.get_dummies(data['Segment'], drop_first=(True))

#  creating dummies 
corporate = pd.get_dummies(data['Segment'], drop_first=( True))
#  Droping the segment column
data.drop(['Segment'], axis = 1 , inplace = True)
data.head(10)
 
#adding the created inteager column
pd.concat([data, corporate], axis = 1)
data.head(10)

#   Checking different values in column "Category"
print(data['Category'].unique())

#   3 category column , converting them in inteager by creating the dummies 

Category = pd.get_dummies(data['Category'], drop_first=(True))

# dropping the 'Category the column
data.drop(['Category'],axis = 1, inplace = True)
data.head(10)

#adding the inteager column

data = pd.concat([data, Category],axis = 1)
data.head(10)

# Country column checking the unique values
print(data['Country'].unique())

#dropping the country column

data.drop(['Country'],axis = 1 , inplace = True)
data.head(5)


#  correlation between 'Sales' column with others

data[data.columns[1:]].corr()['Sales'][:]


#### THANK YOU ####





