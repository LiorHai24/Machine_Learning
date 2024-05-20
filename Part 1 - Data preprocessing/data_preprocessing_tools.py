# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values# the features or independent variables
y = dataset.iloc[:, -1].values# the dependent variable(what you want to predict)
print(X)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')#replace the nan(empty values) by the average('mean')
imputer.fit(X[:, 1:3])#fit this method to our X
X[:, 1:3] = imputer.transform(X[:, 1:3])# apply it
print(X)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')#transformers(ecoder, method, index of column), reminder will remain the other column, without it will be returned only the applied columns.
X = np.array(ct.fit_transform(X))# apply the transform on X(and turn it to np array back)
print(X)
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)#turn the yes/no to binary
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)#taking randomly 20%for text 80% for train, random state = 1 should match the random values picked
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
#the feature scaling method is used when there is a use of information that is not dummy variable or boolean parameter, to match the 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])# set the scale to match the train set and apply
X_test[:, 3:] = sc.transform(X_test[:, 3:])# apply the scale that was set on the train set for the test set
print(X_train)
print(X_test)