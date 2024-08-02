# Support Vector Machine (SVM)

#finding the maximum margin hyperplane that best separates the two classes according to the closest points to the margin that are equal distance from the margin
#will choose the points that are closest to the other category

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


# Training the SVM model on the Training set


# Predicting a new result

# Predicting the Test set results

# Making the Confusion Matrix

# Visualising the Training set results

# Visualising the Test set results
