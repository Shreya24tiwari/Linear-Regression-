# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:07:57 2020

@author: adars
"""


# import exploration files 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns


file_path = '/covid_19_data.csv'

# read in data 
data = pd.read_csv(file_path)


############################################################################## 
#Data Exploration
##############################################################################

#rows and columns returns (rows, columns)
data.shape

#returns the first x number of rows when head(num). Without a number it returns 5
data.head()

#returns the last x number of rows when tail(num). Without a number it returns 5
data.tail()

#returns an object with all of the column headers 
data.columns

#basic information on all columns 
data.info()

#gives basic statistics on numeric columns
data.describe()

#shows what type the data was read in as (float, int, string, bool, etc.)
data.dtypes

#shows which values are null
data.isnull()

data.isnull().sum()

#shows which columns have null values
data.isnull().any()

#shows for each column the percentage of null values 
data.isnull().sum() / data.shape[0]

#plot histograms for all numeric columns 
data.hist() 


############################################################################## 
#Data Manipulation
##############################################################################

# rename columns 
data = data.rename(columns={'Last Update':'Last_Update'})
data.columns

# view all rows for one column
data.Last_Update
data['Last_Update']

# multiple columns by name
data[['Last_Update','Confirmed']]
#data.loc[:['Last_Update','Confirmed']]

#columns by index 
data.iloc[0:4]

#columns in iloc means they answer by a number
data.iloc[0:3,0:2]
#THE FIRST ONE IS FOR ROW (3) AND SECOND ONE IS FOR COLUMN(2)
data.iloc[:,:2]
data.iloc[:,: -1]

data.iloc[0:6,[0,2]]

#columns in loc means they answer by a name
data.loc[0:3]


#Count the unique start Locations
#data["START*"].value_counts()


# drop columns 
#data.drop('ObservationDate', axis =1) #add inplace = True to do save over current dataframe

#drop multiple 

data = data.drop(['ObservationDate'],axis=1)
data.head()

data = data.drop(['SNo','Province/State','Country/Region','Last_Update'], axis =1)
data.head()

data

#correlation 
data.corr()

#Count the unique location
data["Deaths"].value_counts().head()

#Count the unique start Locations
#data["START*"].value_counts()


# to find the correlation among variables (Multicollinearity)
# ----------------------------------------------------------
data.corr()
cor = data.iloc[:,0:].corr()
print(cor)
data.dtypes
data.columns
data
#################################################################

#lambda function 
#data.apply(lambda x: x.colname**2, axis =1)

# pivot table 
#pd.pivot_table(data, index = 'col_name', values = 'col2', columns = 'col3')

# merge  == JOIN in SQL
#pd.merge(data1, data2, how = 'inner' , on = 'col1')

# write to csv 
#data.to_csv('data_out.csv')


##################################################################



   ###LINEAR REGRESSION###




##################################################################



# split the dataset into train and test
# --------------------------------------
train, test = train_test_split(data, test_size = 0.1)
print(train.shape)
print(test.shape)

# split the train and test into X and Y variables
# ------------------------------------------------
train_x = train.iloc[:,0:2]; train_y = train.iloc[:,2]
test_x  = test.iloc[:,0:2];  test_y = test.iloc[:,2]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

train_x.head()
train_y.head()

train.head()

# ensure that the X variables are all numeric for regression
# ----------------------------------------------------------
train.dtypes
    

# To add the constant term A (Y = A + B1X1 + B2X2 + ... + BnXn)
# Xn = ccomp,slag,flyash.....
# ----------------------------------------------------------
lm1 = sm.OLS(train_y, train_x).fit()

# Prediction
# -----------------
pdct1 = lm1.predict(test_x)
print(pdct1)


# store the actual and predicted values in a dataframe for comparison
# -------------------------------------------------------------------
actual = list(test_y.head(5))
type(actual)

predicted = np.round(np.array(list(pdct1.head(5))),2)
print(predicted)
type(predicted)

# Actual vs Predicted
#-----------------------
df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})
print(df_results)


#To Check the Accuracy:
#-----------------------------
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  
