# -*- coding: utf-8 -*-
"""
@author: 肖
"""

'''Logistics回归的原理:
利用Logistics回归进行分类的主要思想是：根据现有数据对分类边界线建立回归公式，
以此进行分类。这里的“回归” 一词源于最佳拟合，表示要找到最佳拟合参数集。
1.找一个合适的预测函数，一般表示为h函数，该函数就是我们需要找的分类函数，它用来预测输入数据的判断结果。
这个过程是非常关键的，需要对数据有一定的了解或分析，知道或者猜测预测函数的“大概”形式，比如是线性函数还是非线性函数。
2.构造一个Cost函数（损失函数），该函数表示预测的输出（h）与训练数据类别（y）之间的偏差，可以是二者之间的差（h-y）
或者是其他的形式。综合考虑所有训练数据的“损失”，将Cost求和或者求平均，记为J(θ)函数，表示所有训练数据预测值与实际类别的偏差。
3.显然，J(θ)函数的值越小表示预测函数越准确（即h函数越准确），所以这一步需要做的是找到J(θ)函数的最小值。
找函数的最小值有不同的方法，Logistic Regression实现时有梯度下降法（Gradient Descent）。'''

import numpy as np
import sklearn.datasets as datasets   # 导入数据进行实验
from sklearn.linear_model import LogisticRegression  # 导入Logistics回归

# 1.获取数据
digits = datasets.load_digits()
x_data = digits.data   # 数据
y_target = digits.target  # 结果

import matplotlib.pyplot as plt
x_data[6].reshape(8,8)
plt.figure(figsize=(2,2))
plt.imshow(x_data[6].reshape(8,8))

# 2.拆分数据
from sklearn.model_selection import train_test_split  # 导入拆分包
X_train,x_test,y_train ,y_test = train_test_split(x_data,y_target,test_size=0.1)

# 3.1 使用logistic回归
logistic = LogisticRegression()
logistic.fit(X_train,y_train)  # 进行训练
logistic.score(x_test,y_test)  # 输出训练的准确率

# 3.2 使用Knn近邻算法进行回归---与上进行对比
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(x_test,y_test)  # 可以看出Knn精度更高

'''
logistic回归实现简单，运算速度快，耗时短，计算代价低;但预测精度较低
KNN近邻算法运算速度慢但是精度更高'''



'''作图比较'''
# 1.重新获取数据
import numpy as np
from sklearn.linear_model import LogisticRegression  # 导入 Logistic算法
from sklearn.neighbors import KNeighborsClassifier   # 导入 KNN算法
import sklearn.datasets as datasets  # 导入数据进行实验
x_data,y_target = datasets.make_blobs(n_samples= 200,centers=4)  # n_samples=200创建200个点(默认100),centers=4表示获取4类点
x_data.shape
y_target.shape

# 画第一张图:
plt.scatter(x_data[:,0],x_data[:,1],c=y_target)  # 由centers=4,y_target会得到4种

# 调用logistic回归
logistic = LogisticRegression()  # 导入Logistic回归
logistic.fit(x_data,y_target)  # 进行训练

# 处理数据---np.meshgrid升维,xx.ravel()将xx进行一维化,np.c_是数组的级联---下面是一系列常用的数据处理方法！
xmin,xmax = x_data[:,0].min(),x_data[:,0].max()  # 第一列最小值和最大值
ymin,ymax = x_data[:,1].min(),x_data[:,1].max()  # 第二列最小值和最大值
x = np.linspace(xmin,xmax,1000)  # 切片
y = np.linspace(ymin,ymax,1000)  # 切片
xx,yy = np.meshgrid(x,y)  # x,y分别进行升维,从1维升成2维
xy = np.c_[xx.ravel(),yy.ravel()]  # 实现点的交叉,xx.ravel()表示将xx进行一维化
y_ = logistic.predict(xy)  # 进行预测

# 绘制第二张图
plt.pcolormesh(xx,yy,y_.reshape(yy.shape))  # y_即颜色,这里绘制的是背影
plt.scatter(x_data[:,0],x_data[:,1],c = y_target,cmap = 'rainbow')  # 绘制点










