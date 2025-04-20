import random

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
random.seed(42)

def genrateDataset(m):
    X = np.random.randn(m)*10
    noise = np.random.randn(m)
    y = 3*X + 1 + 5*noise
    return X,y

def plotDataset(X,y,color = "black",title="Data"):
    plt.title(title)
    plt.scatter(X,y, c = color)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()

def normalizeData(X):
    X = (X - X.mean()) / X.std()
    return X

def trainTestSplit(X, y , train = 0.8 ):
    m = X.shape[0]
    data = np.zeros((m,2))

    data[:,0] = X
    data[:,1] = y

    np.random.shuffle(data)
    split = int(m*train)

    XTrain = data[:split,0]
    yTrain = data[:split,1]

    Xtest = data[split:,0]
    yTest = data[split:,1]

    return XTrain,yTrain, Xtest, yTest


def hypothesis(X,theta):
    return theta[0] + theta[1]*X

def error (X,y,theta):
    m = X.shape[0]
    e = 0
    for i in range(m):
        y_i = hypothesis(X[i], theta)
        e = e + (y[i]-y_i)**2

        return e/(2*m)

def gradient(x,y,theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        exp = hypothesis(X[i],theta) - y[i]
        grad[0] += exp
        grad[1] += exp * X[i]
    return grad/m


def train(X,y, learning_rate = 0.1):
    theta = np.zeros((2,))
    error_list = []
    for i in range(100):
        grad = gradient(X,y,theta)
        error_list.append((error(X,y, theta)))
        theta[0]= theta[0] - learning_rate*grad[0]
        theta[1]= theta[1] - learning_rate*grad[1]
    plt.plot(error_list)
    plt.show()
    return theta

def predict(X,theta):
    return hypothesis(X, theta)

def r2Score(y,yp):
    ymean = y.mean();
    num = np.sum((y - yp)**2)
    denum = np.sum((y -ymean)**2)

    return 1 - (num/denum)




X,y = genrateDataset(100)
print("before normalization of data:")
plotDataset(X,y)
print("after normalization of data:")
X=normalizeData(X)
plotDataset(X,y)
XTrain,yTrain, Xtest, ytest = trainTestSplit(X,y)
print(XTrain.shape,yTrain.shape)
theta = train(X,y)
yp = predict(Xtest,theta)
print(r2Score(ytest,yp))


