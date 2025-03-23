# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: K.HEMANATH
RegisterNumber: 21223100012  
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:, : -1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy

confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
## Output:
Data:
![1](https://github.com/user-attachments/assets/e0293d54-c434-4aab-a94b-1b8ac701d236)

Transformed Data:
![2](https://github.com/user-attachments/assets/b163f669-6302-4aeb-9f18-aca5d5e7f538)

X And Y Values:
![4](https://github.com/user-attachments/assets/4f3c8742-3f8a-4527-9339-5218dc736855)
![5](https://github.com/user-attachments/assets/2cd53514-d3f4-4b07-b0c1-84e131272824)

Accuracy:
![7](https://github.com/user-attachments/assets/df86d571-7d12-4bcc-9e97-b87c0b4fea79)

Classification:
![9](https://github.com/user-attachments/assets/b5649467-a96d-4faf-9e6e-00ddccf4b62d)

Prediction:
![10](https://github.com/user-attachments/assets/602fcb95-bfad-4f21-b76e-2ebbcb39a280)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
