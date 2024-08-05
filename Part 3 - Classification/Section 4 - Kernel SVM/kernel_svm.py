# Kernel SVM
#a solution for a data that is not linearly separable
#the idea: to add another dimention that will make our data linearly separable
#problem: the new dimention can be very complex to compute

#solution: use the kernel trick
#the kernel trick: the idea of adding a dimention without actually adding it
#sigma = the radius of the gaussian sphere (of the inside category)
#the new data point will be categorized accorring to the points that are inside the sphere,
#which means that they have a high value of the added dimention.
#can use multiply kernels.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set


# Feature Scaling


# Training the Kernel SVM model on the Training set


# Predicting a new result

# Predicting the Test set results

# Making the Confusion Matrix


# Visualising the Training set results


# Visualising the Test set results
