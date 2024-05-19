# Multiple Linear Regression
#
# i'm using the method of backwards elimination for this model of multiple linear regression
# 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #all the columns exept the last one(independent variables)
y = dataset.iloc[:, -1].values #last column (independent variable)
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')#transformers(ecoder, method, index of column), reminder will remain the other column, without it will be returned only the applied columns.
X = np.array(ct.fit_transform(X))# apply the transform on X(and turn it to np array back)
print("--- After one hot encoding the countries ---")
print (X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)#taking randomly 20%for text 80% for train, random state = 1 should match the random values picked

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_prediction = regressor.predict(X_test)
np.set_printoptions(precision=2)#show 2 digits after the dot.
#print the them values next to each other
print(np.concatenate((y_prediction.reshape(len(y_prediction),1) , y_test.reshape(len(y_test),1)),axis=1))#reshape it to show in a column, so y_rediction rows with 2 value each row(pred actual)
                                                                                                        #axis = 0 if you want vertical concatenation
                                                                                                        #       1 if you want horizontal concatenation