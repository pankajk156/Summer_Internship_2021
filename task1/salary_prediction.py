#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.externals import joblib

dataset = pd.read_csv('SalaryData.csv')
x = dataset['YearsExperience']
x = x.values.reshape(30,1)
y = dataset['Salary']
model = LinearRegression()
model.fit(x , y)

#saving the model for further usage
from sklearn.externals import joblib
joblib.dump(model, 'salary_prediction.pk1')

#Using the saved model for salary prediction

from sklearn.externals import joblib
salary_model=joblib.load("salary_prediction.pk1")

pred = int(input("Enter Year of Experience : "))
sal = model.predict([[pred]])
print("Predicted Salary : ", sal )