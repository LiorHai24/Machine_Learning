# Decision Tree Regression
#according to a graph with x,y(independent variable)and z(dependent variable)
#dividing the graph with splits, and each side of the split is a branch of the tree
#according to the splits, we will calculate the average of the dependent variable(z) for a given x,y point
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset


# Predicting a new result

# Visualising the Decision Tree Regression results (higher resolution)
