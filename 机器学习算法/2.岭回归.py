k'l# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:49:31 2018

@author: 肖
"""

'''1.普通线性回归'''
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.datasets as datasets

diabetes = datasets.load_diabetes()  # 获取糖尿病数据
x_data = diabetes.data   # 数据
y_target = diabetes.target  # 结果

X_train = x_data[:,2,np.newaxis]  # 获取第二列进行研究,np.newaxis应该是多加1维使结果成为442*1的numpy.ndarray
type(X_train)

from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X_train,y_target,test_size = 0.05)
# X_train,y_train分别使训练数据与训练数据的结果,x_test,y_test分别是测试数据及其结果

lrg = LinearRegression()  # 创建数学模型---使用普通线性回归
lrg.fit(X_train,y_train)  # 进行训练

y_ = lrg.predict(x_test)  # 预测
print(y_)
print(y_test)  # 对比结果

import matplotlib.pyplot as plt  # 画出线性回归结果
plt.scatter(x_test,y_test)  # x与y使一一对应的
plt.plot(x_test,y_,'green')



'''2.矩阵运算--求矩阵的秩、逆矩阵、矩阵的积'''
# 2.1 矩阵运算
x_data.shape   # 表示442个样本,10个特征
x_data.reshape(10,442).shape   # 表示10个样本,442个特征

X_train = np.array([[3,3.5,2],[3.2,3.6,3],[6.6,7,4]])  
np.linalg.matrix_rank(X_train)  # 求矩阵的秩
np.linalg.inv(X_train)  # 求逆矩阵
y_train = np.array([1,2,3])
np.dot(X_train,y_train)  # 求矩阵的积
# 求矩阵解线代  
# 3x+3.2y=118.4
# 3.5x+3.6y=1 则：

price = np.array([[3,3.2],[3.5,3.6]])
total_price = np.array([118.4,135.2])
price_ = np.linalg.inv(price)  # 求解price的逆矩阵
np.dot(price_,total_price)  # 求矩阵的积，这也是求线代解



'''3.岭回归：
1.岭回归可以解决特征数量比样本量多的问题
2.岭回归作为一种缩减算法可以判断哪些特征重要或者不重要，有点类似于降维的效果
3.缩减算法可以看作是对一个模型增加偏差的同时减少方差
岭回归用于处理下面两类问题：
1.数据点少于变量个数
2.变量间存在共线性（最小二乘回归得到的系数不稳定，方差很大）'''

import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.datasets as datasets

# 岭回归实例1：
# 1. 10个方程，但是有30个未知数，显然这不能用线性回归求解
X_train = np.random.randint(0,100,size = (10,30))
y_train = np.random.randint(0,10,size = 10)

# 2.使用普通线性回归---只是试验以下
lrg = LinearRegression()  # 调用普通线性回归
lrg.fit(X_train,y_train)  # 进行训练
x_test = X_train[::2]  # 每隔两行取1行数据,得到5行数据
y_test = y_train[::2]  # 每隔两行取1行数据,得到5行数据

lrg.predict(x_test)  # 进行预测
print(y_test)  # 进行比较

# 3.使用岭回归
from sklearn.linear_model import Ridge  # 导入岭回归
ridge = Ridge(alpha=1)  # 默认alpha=1,表示加入的二阶正则项的最小二乘(最小二乘就是线性回归),alpha=0时就是线性回归
ridge.fit(X_train,y_train)
ridge.predict(x_test)
print(y_test)  # 进行比较 

lrg.coef_  # 得到lrg的系数,也就是这30个数据的系数,可以看出哪个重要哪个不重要。
ridge.coef_  # 同上,得到ridge的系数,可以认为这30个系数的权重


# 岭回归实例2---绘制岭迹线
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# 设置一个十元一次方程---这里涉及到一维的矩阵和二维矩阵相加的广播机制
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])  # np.newaxis（多维数组）增加一个轴

# 设置方程的结果值
y = np.ones(10)

n_alphas = 200  # 设置200个alpha---->生成200个 coef_ 比重
alphas = np.logspace(-10, -2, n_alphas)  # 10^-10 ~ 10^-2 之间选取200个数,np.logspace即log10(x)
clf = linear_model.Ridge(fit_intercept=False)  # intercept表示截距,fit_intercept=False表示不考虑截距的计算,得出的结果是相对值

coefs = []
for a in alphas:
    clf.set_params(alpha=a)   # 动态设置alpha
    clf.fit(X, y)  # 进行训练
    coefs.append(clf.coef_)

# Display results
plt.figure(figsize=(12,9))
#获取当前的画面
#get current axes
ax = plt.gca()


ax.plot(alphas, coefs)
ax.set_xscale('log')  # 设置坐标轴刻度显示的单位 log

#limit xmin xmax 坐标刻度进行了反转
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()





















