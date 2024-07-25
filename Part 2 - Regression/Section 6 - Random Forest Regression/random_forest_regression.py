# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)#number of trees
regressor.fit(X,y)
# Predicting a new result
print(regressor.predict([[6.5]]))
# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)#taking the indicated float values instead of only the the integers to get a smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color='red')#real data points
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')#the points(X_grid) and the prediction
plt.title('Truth or Bluff (Random Forest regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()