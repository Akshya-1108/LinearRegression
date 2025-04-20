import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def normalize (X):
    return (X - X.mean(axis=0))/X.std(axis=0)

def hypothesis(X,theta):
    return np.dot(X,theta)

def loss (X,y,theta):
    yp = hypothesis(X,theta)
    error = np.mean((y-yp)**2)
    return error

def gradient(X,y,theta):
    m = X.shape[0]
    yp = hypothesis(X,theta)
    grad = np.dot(X.T, (yp-y))
    return grad/m

# def preprocess(X):
#     m=X.shape[0]
#     ones = np.ones((m,1))
#     x = np.hstack((ones,X))
#     return X

def preprocess(X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    m = X.shape[0]
    ones = np.ones((m, 1))
    X = np.hstack((ones, X))
    return X

def train(X,y,learning_rate = 0.1, max_iters = 100):
    n = X.shape[1]
    theta = np.random.randn(n)
    error_list = []
    for i in range(max_iters):
        e = loss(X,y,theta)
        error_list.append(e)
        grad = gradient(X,y,theta)
        theta = theta - learning_rate*grad
    plt.plot(error_list)
    return theta

def r2score(y,yp):
    ymean = y.mean()
    num = np.sum((y-yp)**2)
    denum = np.sum((y-ymean)**2)
    return (1-num/denum)



X,y = make_regression(n_samples = 500, n_features = 10, n_informative = 5, noise = 0.8, random_state = 0)
print(X.shape,y.shape)
print(pd.DataFrame(X).head(100))
X= normalize(X)
print(pd.DataFrame(X).head(100))

#visualize the data
for f in range(0,10):
    plt.subplot(4,3,f+1)
    plt.scatter(X[:,f],y)
plt.show()

# Train Test split
Xtrain ,Xtest  , ytrain, ytest = train_test_split(X,y, test_size=0.3, shuffle=False,random_state=0)
# print(Xtrain.shape , ytrain.shape)
# print(Xtest.shape , ytest.shape)


Xtrain = preprocess(Xtrain)
Xtest = preprocess(Xtest)
print(Xtrain.shape , Xtest.shape)
theta = train(Xtrain,ytrain)
yp = hypothesis(Xtest,theta)
print(pd.DataFrame(yp).head(100))
print(r2score(ytest,yp))
