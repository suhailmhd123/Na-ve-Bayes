# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:44:36 2024

@author: H P
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the Titanic dataset
raw_data = pd.read_csv(r"C:\SUHAIL\python data\titanic.csv")

# Display the first few rows of the dataset
raw_data.head()

# Drop the unnecessary columns
data = raw_data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis='columns')
data.head()

# Dealing with missing values
mv = data.isnull().sum()
mv

# Remove all the data that has missing values
data_no_mv = data.dropna(axis=0)
data_no_mv.describe(include='all')

data_with_dummies = pd.get_dummies(data_no_mv, drop_first=True)
data_with_dummies.head()


# Assuming data_with_dummies is your DataFrame with dummy variables
corr_matrix = data_with_dummies.corr().round(2)  # Round to 2 decimal places
sns.heatmap(data=corr_matrix, annot=True)  # Set annot = True to print the values inside the squares
plt.show()

# Drop the 'Fare' column
data_no_multicollinearity = data_with_dummies.drop('Fare', axis=1)
data_no_multicollinearity.head()

# Plot the distribution of 'Age'  (distribution plot)
sns.distplot(data_no_multicollinearity['Age'])
plt.show()

# Declaring the features and the label
features = data_no_multicollinearity.drop('Survived', axis=1)
label = data_no_multicollinearity['Survived']

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and test sets, in a 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# training the model on training set

from sklearn.naive_bayes import GaussianNB
# Build and fit the model
clf = GaussianNB()
clf.fit(X_train, y_train)

# Making predictions
pred = clf.predict(X_test)
pred
# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))
pd.crosstab(y_test,pred, rownames = ['Actual'], colnames= ['Predictions'])

from sklearn.metrics import classification_report
# Assuming y_test and pred are defined
report = classification_report(y_test, pred)
print("Classification Report:\n", report)

## after model fit set a input data is firt class,age(23),male(True)
X_test
clf.predict([[1,23.0,True]])