# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Salary dataset using pandas.

2. Encode the categorical “Position” column using LabelEncoder.

3. Split the data into training and testing sets.

4. Train & Predict using DecisionTreeRegressor and visualize the tree. 

## Program:


### Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
---
```
Developed by: LAKSHMIDHAR N 
RegisterNumber: 212224230138
```
---
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
```

```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```

```py
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()

```

```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```

```py
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("R2 Score = ",r2)
```

```py
dt.predict([[5,6]])
```

```py
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(
    dt,
    feature_names=["Position", "Level"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Salary Prediction", fontsize=14)
plt.show()
```


## Output:

![OUTPUT](image.png)

![OUTPUT](image-1.png)

![OUTPUT](image-2.png)

![OUTPUT](image-3.png)

![OUTPUT](image-5.png)

![Decision tree](image-6.png)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
