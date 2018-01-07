
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#check for working directory
mydir =  (os.getcwd())
print (os.listdir(mydir))

#read dataset
dataset = pd.read_csv("50_Startups.csv")

#check for missing values
print (dataset.isnull().any().any())
dataset.isnull().any()

#splitting the dataset
X=dataset.iloc[:, : 4].values
y = dataset.iloc[:, -1:].values


#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:,3]= labelencoder_X.fit_transform(X.iloc[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X= onehotencoder.fit_transform(X).toarray()

#dumping the dummy variable
X = X[:, 1:]

#splitting the database
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state =0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test results
y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis=1)

X_opt = X[:, [0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()

regressor_OLS.summary()
#---------------------------
X_opt = X[:, [0,1,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()

regressor_OLS.summary()

#---------------------------
X_opt = X[:, [0,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()

regressor_OLS.summary()

#---------------------------

X_opt = X[:, [0,3,5]]

regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()

regressor_OLS.summary()

#---------------------------

X_opt = X[:, [0,3]]

regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()

regressor_OLS.summary()
