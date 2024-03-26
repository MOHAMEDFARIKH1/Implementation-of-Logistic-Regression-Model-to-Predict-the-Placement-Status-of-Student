# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: H MOHAMED FARIKH
RegisterNumber: 212223080032
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
data1
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
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy Score:",accuracy)
print("\nConfusion Matrix:\n",confusion)
print("\nClassification Report:\n",cr)
model.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot 2024-03-26 113026](https://github.com/MOHAMEDFARIKH1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/160568234/40d84be8-f031-4018-acbe-45d13e355df6)
![Screenshot 2024-03-26 113041](https://github.com/MOHAMEDFARIKH1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/160568234/ad36cc36-a102-4b84-b7f1-62da3268a712)
![Screenshot 2024-03-26 113054](https://github.com/MOHAMEDFARIKH1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/160568234/dafc7caa-fedb-4623-acd4-8104e5abc3fb)
![Screenshot 2024-03-26 113106](https://github.com/MOHAMEDFARIKH1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/160568234/f26d27b6-5fb5-4fbe-8415-32ef4d6d2098)
![Screenshot 2024-03-26 113122](https://github.com/MOHAMEDFARIKH1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/160568234/8d9fb168-5df6-4240-8130-faa4f33b5e67)
![Screenshot 2024-03-26 113137](https://github.com/MOHAMEDFARIKH1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/160568234/fd96f7e0-ed9a-4076-a612-320b4b6deb44)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
