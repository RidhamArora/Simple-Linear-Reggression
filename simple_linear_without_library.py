# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

x_train=(X_train-X_train.mean())/X_train.std()
x_test = (X_test-X_test.mean())/X_test.std()
y_train=(X_train-X_train.mean())/X_train.std()
y_test=(Y_test-Y_test.mean())/Y_test.std()
plt.scatter(X_train,Y_train)
plt.show()
plt.scatter(x_train,y_train)
plt.show()

##Gradient Algorithm
def hypothesis(x_train,theta):
    return theta[0]+theta[1]*x_train

def error(X,Y,theta):
    
    m=x_train.shape[0]
    error=0
    
    for i in range(m):
        hx=hypothesis(x_train[i],theta)
        error+=(hx-y_train[i])**2
    
    return error
    
def gradient(x_train,y_train,theta):
    
    grad = np.zeros((2,))
    m=x_train.shape[0]
    
    for i in range(m):
        hx=hypothesis(x_train[i],theta)
        grad[0] = hx-y_train[i]
        grad[1] = (hx-y_train[i])*x_train[i]
    return grad
    
###Algorithm
def gradientDescent(x_train,y_train,learning_rate=0.001):
    theta=np.ones((2,))
    error_list=[]
    itr = 0
    max_itr=1000
    e=1000
    while(e>0.01):
        grad=gradient(x_train,y_train,theta)
        e=error(x_train,y_train,theta)
        error_list.append(e)
        theta[0]=theta[0]-grad[0]*learning_rate
        theta[1]==theta[1]-grad[1]*learning_rate
        itr+=1
    print(itr)    
    return theta,error_list

final_theta,error_list=gradientDescent(x_train,y_train)
plt.plot(error_list)
plt.show()
plt.scatter(X_train,Y_train,color='Blue')
plt.scatter(X_test,Y_test,color='Green')
y_pred=hypothesis(x_test,final_theta)
Y_pred=y_pred*Y_test.std()+Y_test.mean()
plt.plot(X_test,Y_pred,color='Orange')
plt.show()
