import numpy as np
import matplotlib.pyplot as plt

def read_data(filepath):
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = [], [], [], [], [], []
    
    # extract data from filepath
    with open(filepath, 'r') as file:
        for i in range(100):
            line = file.readline().strip()
            if not line:
                break
            features = list(map(float, line.split()))
            Xtrain.append(features[:-1])
            Ytrain.append(features[-1])
        for i in range(100, 150):
            line = file.readline().strip()
            if not line:
                break
            features = list(map(float, line.split()))
            Xval.append(features[:-1])
            Yval.append(features[-1])
        for i in range(150, 200):
            line = file.readline().strip()
            if not line:
                break
            features = list(map(float, line.split()))
            Xtest.append(features[:-1])
            Ytest.append(features[-1])

    Xtrain, Ytrain = np.array(Xtrain), np.array(Ytrain)
    Xval, Yval = np.array(Xval), np.array(Yval)
    Xtest, Ytest = np.array(Xtest), np.array(Ytest)
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def normal_equation(Xtrain, Ytrain):
    XT_X = Xtrain.T @ Xtrain
    XT_y = Xtrain.T @ Ytrain
    return np.linalg.inv(XT_X) @ XT_y

def polynomial_features(X):
    bias = X[:, [0]]
    x1   = X[:, [1]]
    x2   = X[:, [2]]
    return np.hstack([bias, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])

filepath = 'regression-data.txt'
Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = read_data(filepath)

#linear
w_best = normal_equation(Xtrain, Ytrain)
print("Linear Mean Squared Error:", mean_squared_error(Ytest, Xtest.dot(w_best)))

#poly
XtrainPoly = polynomial_features(Xtrain)
w_best_poly = normal_equation(XtrainPoly, Ytrain)
XtestPoly = polynomial_features(Xtest)
print("Polynomial Mean Squared Error:", mean_squared_error(Ytest, XtestPoly.dot(w_best_poly)))

answer = {
    "Linear Mean Squared Error": '0.7194635198584615',
    "Polynomial Mean Squared Error": '0.752509436607882'
}