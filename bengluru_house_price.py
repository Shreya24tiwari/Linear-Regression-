# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:05:56 2020

@author: adars
"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import array as arr
from sklearn import svm
from sklearn.metrics import confusion_matrix
#from pandas_ml import ConfusionMatrix
from sklearn.metrics import classification_report as cr
from sklearn.linear_model import LogisticRegression


path = "/Bengluru_House_price.csv"
data = pd.read_csv(path)
data
data.head()
data.columns
data.dtypes
data.count

data.describe(include = 'all')
data.describe()


#As we show data bath, price average value is Less than mean value to 
#the graph shows Right Skewed.
#balcony average value is greater than mean value so graph shows Left Skewed.

data.isnull().sum() 


###########################

def change_to_float(area_size):
        if isinstance(area_size, str):
             area_size = area_size.split('Sq.')[0]
             area_size = area_size.split('Perch')[0]
             area_size = area_size.split('Acres')[0]
             area_size = area_size.split('Guntha')[0]
             area_size = area_size.split('Grounds')[0]
             area_size = area_size.split('Cents')[0]
             area_size = area_size.split('-')
             area_size = list(map(float,area_size))
             area_size = sum(area_size)  / len(area_size)
             
        return area_size


###########################################
    
data['total_sqft'] = data['total_sqft'].apply(lambda x : change_to_float(x))

data['total_sqft'].head()
data['total_sqft'] = data['total_sqft'].astype('float64')


size_mode = data['size'].mode()[0]
data.loc[data['size'].isna(), 'size'] = size_mode

data['size'] = data['size'].apply(lambda x: x.split(' ')[0])
data['size'].head()
data['size'] = data['size'].astype('float64')

data.head()
data.corr()
data.columns
data.dtypes


#drop the area_type and availability, location columns
 
data = data.drop(['area_type','availability','location','society'],axis=1)
print(data)

data.head()
data.corr()
data.describe()
data.isnull().sum()


#bath and balcony have missing column so 

data['balcony'].fillna((data['balcony'].mean()), inplace=True)

print(data)


data['bath'].fillna((data['bath']).mean(), inplace=True)
print(data)


#############################################


#LINEAR REGRESSION


# to find the correlation among variables (Multicollinearity)
# ----------------------------------------------------------
data.corr()
cor = data.iloc[:,0:3].corr()
print(cor)
    


# split the dataset into train and test
# --------------------------------------
train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape)

# split the train and test into X and Y variables
# ------------------------------------------------
train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]
test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]

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
#predicted = list(train.head())
#type(predicted)
predicted = np.round(np.array(list(pdct1.head(5))),2)
print(predicted)

# Actual vs Predicted
#-----------------------
df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})
print(df_results)
df_results

#To Check the Accuracy:
#-----------------------------
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  


##############################################
