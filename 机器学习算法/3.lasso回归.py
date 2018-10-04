# -*- coding: utf-8 -*-
"""
@author: 肖
lasso回归:
对于参数w增加一个限定条件，能到达和岭回归一样的效果：即所有权重的绝对值之和<=lambda
在lambda足够小的时候，一些系数会因此被迫缩减到0
"""

import numpy as np

'''1.1 Lasso回归就是一种线性回归---案例1:'''
from sklearn.linear_model import Lasso
import sklearn.datasets as datasets  # 为了导入数据

boston = datasets.load_boston()  # 获得波士顿房价信息
x_data = boston.data  # 获得房价数据
y_target = boston.target  # 获得房价(结果)

from sklearn.model_selection import train_test_split  # 导入分割功能
X_train,x_test,y_train,y_test = train_test_split(x_data,y_target,test_size = 0.05)  # 进行分割
# X_train时训练数据,y_train是训练结果l; x_test是测试数据,y_test是测试结果

lasso_estimator = Lasso(alpha=1)  # 默认alpha=1,当alpha=0时就是普通线性回归
lasso_estimator.fit(X_train,y_train)  # 进行训练
y_ = lasso_estimator.predict(x_test)  # 预测数据
print(y_test)  # 进行比较

'''1.2 下面使用普通的线性回归进行对比'''
from sklearn.linear_model import LinearRegression  # 导入普通线性回归
lrg_estimator = LinearRegression()
lrg_estimator.fit(X_train,y_train)  # 进行训练
y_ = lrg_estimator.predict(x_test)  # 预测



'''2.普通线性回归、岭回归与lasso回归比较'''
from sklearn.linear_model import LinearRegression,Ridge,Lasso  # 分别导入普通线性回归,岭回归,罗斯回归

# 创建数据的样本量 50个,属性 200个
sample = 50
feture = 200
x_data = np.random.randn(sample,feture)  #  np.random.randn生成标准正太分布

coef = np.random.randn(feture)  # 生成系数(其实就是解),让它和x_data做矩阵乘法,这样就获取了y
# 现在想将coef中190个元素设置为0，只剩下10个数，但并没有告诉机器这些就是解
inds = np.arange(0,200)  # 生成1到199的数
np.random.shuffle(inds)  # np.random.shuffle是随机排序的方法
coef[inds[:190]] = 0  # 将其中190个数的元素置为0,其余10个数不为0,这就完成了任务需求了

y_target = np.dot(x_data,coef)  # 得到50*1的矩阵
y_target += 0.01*np.random.randn(sample)  # 添加一些噪声,增加一些难度
# 最终得到的是: x_data * coef = y_target的矩阵乘法

# 进行拆分:
X_train,x_test,y_train,y_test = train_test_split(x_data,y_target,test_size = 0.2)

'''分别使用线性回归，岭回归，Lasso回归进行数据预测:'''
# 1.线性回归的情况:
lrg = LinearRegression()
lrg.fit(X_train,y_train)
lrg.score(x_test,y_test)   #　相当于求lrg.predict(x_test),y_test测试的得分情况（准确率）
lrg.coef_   # 方程的解
lrg.intercept_  # 方程的截距(即常数)

# 2.岭回归的情况:
ridge = Ridge(alpha= 0.7)
ridge.fit(X_train,y_train)
ridge.score(x_test,y_test)  
ridge.coef_  # 方程的解

# 3.lasso回归的情况:(情况应该是最好的)
lasso = Lasso(alpha=0.2)
lasso.fit(X_train,y_train)
lasso.score(x_test,y_test)
lasso.coef_  # 方程的解,可以发现有很多0解; 对比coef有190个0解,这也是效果好的一种说明
lasso.predict(x_test) # 其实就是 np.dot(x_test,lasso.coef_)+lasso.intercept_
np.dot(x_test,lasso.coef_)+lasso.intercept_  # lasso.intercept_是方程的截距(即常数)


'''画图---绘制coef(解)'''
import matplotlib.pyplot as plt
plt.figure(figsize=(12,9))

axes1 = plt.subplot(221)
axes1.plot(coef,label = 'answer')  # 绘制自己创建的coef系数，标准答案
plt.legend()

axes2 = plt.subplot(222)
axes2.plot(lrg.coef_,label = 'linear',color = 'r') 
plt.legend()

axes3 = plt.subplot(223)
axes3.plot(ridge.coef_,label = 'ridge',color = 'b') 
plt.legend()

axes4 = plt.subplot(224)
axes4.plot(lasso.coef_,label = 'lasso',color = 'g') 
plt.legend()











