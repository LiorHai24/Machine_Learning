# Support Vector Regression (SVR)
#Non-Linear
#
#usage of feature scaling is used when there is a connection between the dependent and independent variables, like level and salary.
#
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values#two dimentional array with one column
y = dataset.iloc[:, -1].values#one dimentional array
#print(X)
#print(y)
y = y.reshape(len(y),1)#reshape the y to a two dimentional array(y rows with 1 column)
#print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#print(X)
#print(y)#(-3 to 3)
# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')#radial basis function, the most common kernel
regressor.fit(X, y)
# Predicting a new result
#predicting the X value of 6.5 scaled, and then inverse the scaling to get the real value of y
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)))#the prediction is scaled, so we need to inverse it
# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color='red')#real data points
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')#the poly regression
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)#scaling back the X values
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')#same here
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')#X_grid is not scaled, so we need to scale it
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()