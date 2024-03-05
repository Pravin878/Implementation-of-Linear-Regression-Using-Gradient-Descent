# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries in python required for finding Gradient Design.
2. Read the dataset file and check any null value using .isnull() method.
3. Declare the default variables with respective values for linear regression.
4. Calculate the loss using Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.
7. Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.

## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: pravin kumar G.
RegisterNumber:  212222230109
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header =None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def computeCost(X,y,theta):

  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  j=1/(2*m)* np.sum(square_err)
  return j

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range (num_iters):
    predictions=X.dot(theta)
    error = np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history  

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1" )

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Grading Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict (x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population =70,000,we predict a profit a profit of $"+str(round(predict2,0)))
```

## Output:
### Profit Prediction Graph:
![e3 (1)](https://github.com/S-ARVIND01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707337/4205e43d-52b0-468f-949d-04b92d9d42fa)

### Compute Cost Value:
![ex3-2](https://github.com/S-ARVIND01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707337/b78cdfa6-23c2-4e5b-8270-8ea0192cec6c)

### h(x) Value:
![ex3-3](https://github.com/S-ARVIND01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707337/c5c01bd9-147d-44e4-a263-4a48c6028cd5)

### Cost function using Gradient Descent Graph:
![ex3-4](https://github.com/S-ARVIND01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707337/3cde6a46-f7f3-46b3-b30f-e89af6ec21e2)

### Profit Prediction Graph:
![ex3-5](https://github.com/S-ARVIND01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707337/ccfe7803-3ef4-451e-aa63-5fc94dc91e7c)

### Profit for the Population 35,000:
![ex3-6](https://github.com/S-ARVIND01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707337/ef85e412-53e4-4bf7-af90-715e4a9b0915)

### Profit for the Population 70,000:
![ex3-7](https://github.com/S-ARVIND01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707337/81d7c054-a87b-472c-8aad-2d2475c8fdc7)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
