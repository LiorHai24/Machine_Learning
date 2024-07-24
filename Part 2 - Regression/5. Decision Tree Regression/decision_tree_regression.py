# Decision Tree Regression
#according to a graph with x,y(independent variable)and z(dependent variable)
#dividing the graph with splits, and each side of the split is a branch of the tree
#according to the splits, we will calculate the average of the dependent variable(z) for a given x,y point
#no need for feature scaling
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
#no need to transform because the decision tree regression model is not scaled.
print(regressor.predict([[6.5]]))
# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.1)#taking the values instead of between the int's to get a smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color='red')#real data points
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')#the points(X_grid) and the prediction
plt.title('Truth or Bluff (Decision Tree regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()