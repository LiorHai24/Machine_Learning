# K-Nearest Neighbors (K-NN)
#used on a graph which is diveded into categories
#helps decide which category a new data point belongs to based on the majority of its K(usually 5)-nearest neighbors

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

# Training the K-NN model on the Training set


# Predicting a new result

# Predicting the Test set results

# Making the Confusion Matrix


# Visualising the Training set results


# Visualising the Test set results
