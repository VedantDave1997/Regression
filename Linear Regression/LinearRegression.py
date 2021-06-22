#---------------- SIMPLE LINEAR REGRESSION ----------------#
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#---------------- DATASET ----------------#
'''
Data set:
Independant variable X: Outside Air Temperature
Dependant variable Y: Overall daily revenue generated in dollars
'''

# Importing the dataset
IceCream = pd.read_csv("IceCreamData.csv")  # Read a CSV File
IceCream.head()
IceCream.tail()
IceCream.describe()
IceCream.info()


#---------------- VISUALIZE DATA ----------------#
sns.lmplot(x='Temperature', y='Revenue', data=IceCream)
plt.title('Revenue Generated vs. Temperature')
plt.show()


#---------------- CREATE TESTING AND TRAINING DATASET ----------------#
y = IceCream['Revenue']
X = IceCream[['Temperature']]

# Splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#---------------- TRAIN THE MODEL ----------------#
# Instatiating Object of LinearRegression Class
'''
fit_intercept:
True - Model decides slope 'm' and intercept 'b'
False - Model decides slope 'm' and intercept 'b' is zero by defaut
'''

regressor = LinearRegression(fit_intercept =True)
regressor.fit(X_train,y_train)
print('Linear Model slope (m): ', regressor.coef_)
print('Linear Model intercept (b): ', regressor.intercept_)


#---------------- TEST THE MODEL ----------------#
y_predict = regressor.predict( X_test)

# Visualise the Train Results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand(Training dataset)')
plt.show()


#---------------- VISUALIZE TEST DATA ----------------#
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Hours')
plt.title('Revenue Generated vs. Hours @Ice Cream Stand(Test dataset)')
plt.show()


#---------------- PREDICT DATA ----------------#
x_predict = 20
y_predict = regressor.predict(np.asarray([x_predict]).reshape(-1, 1))
print('Data Predicted on: ', x_predict)
print('Predicted Data: ', y_predict)