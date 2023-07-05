# Binary Classification --> Whether the patient have a heart disease or no
# Disease is represented by 1 and No Disease by 0
# Dataset=https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/download?datasetVersionNumber=2
# Importing the necessary libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv("./HeartDisease/dataset/heart.csv")
# print(data.head()) 
# # --> Look at the data
# print(data.shape) 
# print(data.info())

# --> No Missing Values
# print(data.isnull().sum()) 


# print(data["target"].value_counts())
#  --> 526 With disease 499 without disease

# #Target to predict against
y=data["target"]
# using the Feature
X=data.drop(columns="target",axis=1)
# Straitify --> the binary will be distributed in an even manner
# random_state --> For reproducibilty
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,train_size=0.7,stratify=y,random_state=1)
# print(X.shape,X_train.shape,X_test.shape)


# Without Scaling
# Model training
lr=LogisticRegression()


lr.fit(X_train,y_train)

x_train_pred=lr.predict(X_train)
training_acc=accuracy_score(x_train_pred,y_train)
print("Training Data Accuracy: ",training_acc) 
# Achieved a score of 84.37% 

X_test_pred=lr.predict(X_test)
testing_acc=accuracy_score(X_test_pred,y_test)
print(f"Testing data accuracy: {testing_acc}")
##Achieved accuracy of 86.03%

input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)
# Should predict 1
input_data_np=np.array(input_data)

# Reshape to predict against one instance
reshaped_array=input_data_np.reshape(1,-1)
pred=lr.predict(reshaped_array)
print(pred)
# Predicts 1

if(pred[0]==0):
    print("Person doesn't have a heart disease")
else:
    print("The person have a heart disease")






# # Scaling the data
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline


lr=LogisticRegression()
pipe=make_pipeline(StandardScaler(),LogisticRegression())
pipe.fit(X_train,y_train)
Pipeline(steps=[("standardScaler",StandardScaler()),("logisticregression",LogisticRegression())])
train_score=pipe.score(X_train,y_train)
test_score=pipe.score(X_test,y_test)
print("Train score:", train_score)
print("Testing score: ",test_score)
# Accuracy of 85.06%