# Support Vector Regression (SVR)
#Non-Linear

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
y = y.reshape(len(y),1)
print(y)

# Feature Scaling


# Training the SVR model on the whole dataset

# Predicting a new result

# Visualising the SVR results


# Visualising the SVR results (for higher resolution and smoother curve)
#X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
#plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
#plt.title('Truth or Bluff (SVR)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()