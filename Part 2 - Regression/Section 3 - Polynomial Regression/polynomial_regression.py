# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values #the 2nd column represents the job levels so no need for the first which describes it by name
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset to see the difference
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)#we want to train the regression on the whole data set for accurate value

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)#x1 represents the position levels and y represents the salaries higher degree=more accurate
X_poly = poly_reg.fit_transform(X)
lin_reg_for_poly = LinearRegression()
lin_reg_for_poly.fit(X_poly, y)

# Visualising the Linear Regression results
#plt.scatter(X,y, color='red')
#plt.plot(X, lin_reg.predict(X), color = 'blue')
#plt.title('Truth or Bluff (linear regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()
# Visualising the Polynomial Regression results
plt.scatter(X,y, color='red')#real data points
plt.plot(X, lin_reg_for_poly.predict(X_poly), color = 'blue')#the poly regression
plt.title('Truth or Bluff (polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
print("The prediction of our example with the linear regression is" , lin_reg.predict([[6.5]]))#two squared brackets because this is a 2 dimentional graph
# Predicting a new result with Polynomial Regression
#the input to the predict needs to be x values to the power needed in the poly method, predicting according to the poly regressor, to fit for the value 6.5
print("The prediction of our example with the polynomial regression is", lin_reg_for_poly.predict(poly_reg.fit_transform([[6.5]])))