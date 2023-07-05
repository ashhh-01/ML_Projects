import pandas as pd
import numpy as np
import seaborn as sm
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data=pd.read_csv("./LoanPrediction/Data/train_u6lujuX_CVtuZ9i (1).csv")
# print(data.head())
# print(data.info())

# print(data.shape)
# The data contains 614 rows and 13 columns

# The data have null values - total 149 missing values across the dataset
# print(data.isna().sum())
newData=data.dropna()
# No missing Values
# print(newData.isna().sum())
newData.replace({"Loan_Status":{"N":0,"Y":1}},inplace=True)
# inplace-->replace all
# We can use one-hot-encoding or Label-encoding  or pandas.get_dummies
# from sklearn.preprocessing import LabelEncoder

# ens=LabelEncoder()
# newData.Loan_Status=ens.fit_transform(newData.Loan_Status)
# print(newData["Loan_Status"].unique())

# print(newData["Loan_Status"].head())

# print(newData["Dependents"].value_counts()) 
# Replacing '3+'-->4
newData=newData.replace(to_replace="3+",value=4)
# print(newData["Dependents"].value_counts()) 

# Visualisation
# Education Vs Loan Status
# sm.countplot(x="Education",hue="Loan_Status",data=newData)
# plt.show()

# Married vs Loan_status
# sm.countplot(x="Married",hue="Loan_Status",data=newData)
# plt.show()

# Converting all categorical columns to Numerical values
newData.replace({"Married":{"No":0,"Yes":1},"Gender":{"Male":1,"Female":0},"Self_Employed":{"No":0,"Yes":1},
                         "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2},"Education":{"Graduate":1,"Not Graduate":0}},inplace=True)
# print(newData.head())

# Separating Data
X=newData.drop(columns=["Loan_ID","Loan_Status"],axis=1)
y=newData["Loan_Status"]
# print(X.shape)
# print(y.shape)

# Train & Test
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,stratify=y,test_size=0.3)
# print(X_test.shape,X_train.shape)
# Support vector classifier --> Will find the hyperplane
classifier=svm.SVC(kernel="linear")
# Train
classifier.fit(X_train,y_train)
# X_train_acc=classifier.predict(X_train)
# print("Train accuracy: ",accuracy_score(X_train_acc,y_train))
# Accuracy of 77.67%

X_test_acc=classifier.predict(X_test)
print("Test accuracy: ",accuracy_score(X_test_acc,y_test))
# Test Accuracy of 77.638%
