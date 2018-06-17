# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#read years of experience into X (Independent variable)
X = dataset.iloc[:, :-1].values
#read salary into Y (dependent variable)
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Simple Linear Regression to the Training set
#import linerregression library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


## Multiple Linear Regression

# Importing the dataset
multdataset = pd.read_csv('50_Startups.csv')
A = multdataset.iloc[:, :-1].values
b = multdataset.iloc[:, 4].values

# Encoding categorical data
# State is a categorical data and is mapped to numeric category
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
A[:, 3] = labelencoder.fit_transform(A[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
A = onehotencoder.fit_transform(A).toarray()

# Avoiding the Dummy Variable Trap
# by removing the constant variable
A = A[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(A_train, b_train)

# Predicting the Test set results
b_pred = regressor.predict(A_test)

