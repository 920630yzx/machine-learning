# -*- coding: utf-8 -*-
"""
对应jupyter：10.knn
"""

import numpy as np
import matplotlib.pyplot as plt
img = plt.imread('G:/数据分析--腾讯视频网络班/day10 KNN/066-KNN原理回归与分类/4.png')
plt.imshow(img)

'''1.近邻算法---用于分类KNeighborsClassifier'''
# 导入机器学习的包
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import pandas as pd
from pandas import Series,DataFrame

'''案例1:'''
# 读取excel中的第二张表,sheetname=1 这样设置也是可以的
movie = pd.read_excel('G:/数据分析--腾讯视频网络班/day10 KNN/066-KNN原理回归与分类/movies.xlsx',sheetname='Sheet2')  
X = movie[['武打镜头','接吻镜头']]             # 获取movie中的两列
y = movie['分类情况']                        # 获取movie中的一列
knn = KNeighborsClassifier(n_neighbors=5)   # n_neighbors设置邻居数量，一般需要设置为奇数，默认值为5
knn.fit(X,y)                           # 进行训练
X_test = np.array([[130,1],[30,2]])    # 写出需要预测的数据
y_ = knn.predict(X_test)               # 进行预测

'''案例2: 样本不均衡，会导致算法预测不准确。例：'''
moive2 = movie.drop([0,1])      # 删除movie1中的头两行
knn = KNeighborsClassifier(3)   # 使用KNN进行分类,近邻个数n_neighbors设置为3个
knn.fit(moive2[['武打镜头','接吻镜头']],moive2['分类情况'])   # 进行训练
X_test = np.array([[130,1],[30,2]])                       # 写出需要预测的数据
knn.predict(X_test)    # 此时预测会不准确，因为爱情片太多，当然都会“投票”给爱情篇

'''案例3: weights='distance' 例：'''
knn = KNeighborsClassifier(2,weights='distance')   # 设置成距离越近权重越大,当两者各为1票时依据越近的占据更大的权重。
knn.fit(moive2[['武打镜头','接吻镜头']],moive2['分类情况'])
X_test = np.array([[130,1],[30,2]])  
knn.predict(X_test)

'''案例4: 例：'''
import sklearn.datasets as datasets   # sklearn库提供一些样本
iris = datasets.load_iris()           # 获取python自带数据
X = iris['data']    # 获取样本的特征
y = iris['target']  # 获取样本的结果
index = np.arange(150)
np.random.shuffle(index)  # 打乱index顺序
# 根据索引对X，和y进行排序
X = X[index]
y = y[index]
X_train = X[:120]  # 150个数据取出来120个作为训练，剩下的30个作为测试，预测数据
X_test = X[120:]   # 使用30个数据作为验证，验证是否好使
y_train = y[:120]
y_test = y[-30:]   # y[120:]也可,取最后30个数据

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)  # 进行训练
y_ = knn.predict(X_test)  # 进行预测
knn.score(X_test,y_test)      # 获得预测准确率
(y_ == y_test).sum()/y_.size  # 获得预测准确率,其实本质是一样的

'''2.近邻算法---用于回归KNeighborsRegressor'''












