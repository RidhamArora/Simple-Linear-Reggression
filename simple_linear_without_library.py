# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 1)

x_train=(X_train-X_train.mean())/X_train.std()
x_test = (X_test-X_test.mean())/X_test.std()
y_train=(Y_train-Y_train.mean())/Y_train.std()
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
    theta[0]=1
    theta[1]=1
    error_list=[]
    itr = 0
    max_itr=1000
    e=1000
    theta_list=[]
    while(itr<max_itr):
        grad=gradient(x_train,y_train,theta)
        e=error(x_train,y_train,theta)
        error_list.append(e)
        theta_list.append((theta[0],theta[1]))
        theta[0]=theta[0]-grad[0]*learning_rate
        theta[1]==theta[1]-grad[1]*learning_rate
        itr+=1
    print(itr)    
    return theta,error_list,theta_list

final_theta,error_list,theta_list=gradientDescent(x_train,y_train)
plt.plot(error_list)
plt.show()
plt.scatter(x_train,y_train,color='Blue')
plt.scatter(x_test,y_test,color='Green')
y_pred=hypothesis(x_test,final_theta)
plt.plot(x_test,y_pred,color='Orange')
plt.show()
plt.scatter(X_train,Y_train,color='Blue')
plt.scatter(X_test,Y_test,color='Green')
y_pred=hypothesis(x_test,final_theta)
Y_pred=y_pred*Y_train.std()+Y_train.mean()
plt.plot(X_test,Y_pred,color='Orange')
plt.show()

#Visualizing Cost function in 3d
T0=np.arange(-2,3,0.01)
T1=np.arange(-2,3,0.01)

T0,T1=np.meshgrid(T0,T1)

J=np.zeros(T0.shape)

m=T0.shape[0]
n=T1.shape[1]

for i in range(m):
    for j in range(n):
        J[i,j]=np.sum((y_train-T1[i,j]*x_train-T0[i,j])**2)

print(J.shape)

theta_list=np.array(theta_list)

fig = plt.figure()
axes=fig.gca(projection='3d')
axes.plot_surface(T0,T1,J,cmap='rainbow',alpha=0.05)
axes.scatter(theta_list[:,0],theta_list[:,1],error_list,color='black',marker='^')
plt.show()

fig = plt.figure()
axes=fig.gca(projection='3d')
axes.contour(T0,T1,J,cmap='rainbow',alpha=0.5)
axes.scatter(theta_list[:,0],theta_list[:,1],error_list,color='red',marker='^')
plt.show()

u=((Y_test-Y_pred)**2).sum()
v= ((Y_test - Y_test.mean()) ** 2).sum()
r=(1-(u/v))
print(r)









