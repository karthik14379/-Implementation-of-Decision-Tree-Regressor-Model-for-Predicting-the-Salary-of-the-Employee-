# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step 1: Load and Explore the Dataset
- Import the dataset using a data analysis library.
- Display the first few rows to understand the structure.
- Check for missing values and basic data types using info methods.

---

### Step 2: Preprocess the Data
- Convert categorical variables (e.g., Position) into numeric form using encoding (e.g., Label Encoding).
- Split the data into:
  - **Features (X)** – Independent variables like Position, Level
  - **Target (y)** – Dependent variable like Salary

---

### Step 3: Split the Data and Train the Model
- Split the dataset into training and testing sets (e.g., 80% train, 20% test).
- Initialize and train a **Decision Tree Regressor** using the training data.

---

### Step 4: Make Predictions and Evaluate the Model
- Predict the output for the test set using the trained model.
- Evaluate the model’s performance using:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² Score (Coefficient of Determination)**
- Use the trained model to make predictions on new/unseen data points.

---


## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: B Surya Prakash
RegisterNumber:  212224230281
*/
```
```python
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
print(df.head())

x=df[["Position","Level"]]
y=df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
print(y_pred)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)
mse

import numpy as np

rmse=np.sqrt(mse)
rmse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
<img width="719" height="138" alt="image" src="https://github.com/user-attachments/assets/167cc4b4-ec46-4b3b-9b02-74f46f41ed88" />

<img width="722" height="154" alt="image" src="https://github.com/user-attachments/assets/abdf004e-c219-4681-a84e-f7606485f1a9" />

<img width="722" height="66" alt="image" src="https://github.com/user-attachments/assets/2f5f81c3-a284-4ed0-b554-20498dc4a44f" />

<img width="713" height="103" alt="image" src="https://github.com/user-attachments/assets/b7cff12a-5119-4791-b9d7-075658678d11" />

## Y_Pred:
<img width="715" height="53" alt="image" src="https://github.com/user-attachments/assets/cc364ffd-f6cf-46a2-81f7-314675eadba1" />

## MSE:
<img width="720" height="25" alt="image" src="https://github.com/user-attachments/assets/ae71b6d6-72f3-44d4-bee9-495d9b888876" />

### RMSE:
<img width="715" height="22" alt="image" src="https://github.com/user-attachments/assets/7ca241f1-56cf-435a-873b-73454b371cad" />

### R2 Score:
<img width="719" height="26" alt="image" src="https://github.com/user-attachments/assets/44150a59-ea96-4ec7-8153-3c046768c51d" />



<img width="727" height="33" alt="image" src="https://github.com/user-attachments/assets/c6970e9a-178f-4f4f-aabc-732f9c4fb5ca" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
