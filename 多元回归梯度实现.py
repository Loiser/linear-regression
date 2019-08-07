# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def linearRegression(data_X,data_Y,learningRate,loopNum):
    W = np.zeros(shape=[1, data_X.shape[1]])
    # W的shape取决于特征个数，而x的行是样本个数，x的列是特征值个数
    # 所需要的W的形式为 行=特征个数，列=1 这样的矩阵。但也可以用1行，再进行转置：W.T
    # X.shape[0]取X的行数，X.shape[1]取X的列数
    b = 0

    #梯度下降
    for i in range(loopNum):
        W_derivative = np.zeros(shape=[1, data_X.shape[1]])
        b_derivative, cost = 0, 0

        WXPlusb = np.dot(data_X, W.T) + b   # W.T：W的转置
        W_derivative += np.dot((WXPlusb - data_Y).T, data_X)  # np.dot:矩阵乘法
        b_derivative += np.dot(np.ones(shape=[1, data_X.shape[0]]), WXPlusb - data_Y)
        cost += (WXPlusb - data_Y)*(WXPlusb - data_Y)
        W_derivative = W_derivative / data_X.shape[0]  # data_X.shape[0]:data_X矩阵的行数，即样本个数
        b_derivative = b_derivative / data_X.shape[0]


        W = W - learningRate*W_derivative
        b = b - learningRate*b_derivative

        cost = cost/(2*data_X.shape[0])
        if i % 100 == 0:
            print(cost)
    print(W)
    print(b)

if __name__== "__main__":
    X = np.random.normal(0, 10, 100)
    noise = np.random.normal(0, 0.05, 20)
    W = np.array([[3, 5, 8, 2, 1]])  #设5个特征值
    X = X.reshape(20, 5)     #reshape成20行5列
    noise = noise.reshape(20, 1)
    Y = np.dot(X, W.T)+6 + noise
    linearRegression(X, Y, 0.003, 5000)