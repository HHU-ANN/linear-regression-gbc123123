# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y = read_date()
    lam = 0.01
    w = np.dot(np.linalg.inv(np.dot(x.T,x)+lam*np.eye(x.shape(1))),np.dot(x.T,y))
    return sum(w*date)
    
def lasso(data):
    x,y = read_date()
    alpha = 1e-10
    step = 1e-9
    item = 1000
    m,n = x.shape
    w = np.zeros(n)
    for i in range(item):
        y_hat = np.dot(x,w)
        dw = np.dot(x.T,(y_hat-y))/m+alpha*np.sign(w)
        w = w-dw*step
    return sum(w*date)


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y