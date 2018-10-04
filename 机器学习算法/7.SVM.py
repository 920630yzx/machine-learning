# -*- coding: utf-8 -*-
"""
@author: 肖
"""

'''Support Vector Machine。支持向量机，其含义是通过支持向量运算的分类器。
其中“机”的意思是机器，可以理解为分类器。 那么什么是支持向量呢？在求解的过程中，
会发现只根据部分数据就可以确定分类器，这些数据称为支持向量。'''

'''1.svm分类案例--基于线性 kernel='linear''''

import numpy as np
from sklearn.svm import SVC  # SVC继承自libsvm
import matplotlib.pyplot as plt

a = [[1,1],[2,2],[3,3]]
b = [[-1,0],[-2,2],[-3,1]]
np.r_[a,b]

# np.random.randn是生成标准正太分布,-[2,2]是得到的点x轴向左移动2格,y轴向下移动2格
X_train = np.array([np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]])
X_train.shape  # (2, 20, 2)
X_train = X_train.reshape((40,2))   # 两组数据放在一起

plt.scatter(X_train[:,0],X_train[:,1])  # 画点图，可以看出会分成两类点

y_train = ['r']*20+['b']*20   # 结果
plt.scatter(X_train[:,0],X_train[:,1],c = y_train)  # 画点图，这次加上颜色 

# 1.进行机器学习训练---这里使用线性kernel='linear'
svc = SVC(kernel='linear')  # 调用是svm,kernel='linear'表示使用线性
svc.fit(X_train,y_train)    # 进行训练
coef_ = svc.coef_             # 提取斜率系数,分别表示y轴系数，x轴系数
w = -coef_[0,0]/ coef_[0,1]   # 斜率结果
intercept_ = svc.intercept_   # 提取截距系数
b = -intercept_[0]/coef_[0,1] # 获得截距，暂不管为什么会用就好

x = np.linspace(-4,4,100)
plt.scatter(X_train[:,0],X_train[:,1],c = y_train)
plt.plot(x,w*x + b)  # y = w*x + b

# 2.画出更完整的图
vectors_ = svc.support_vectors_  # 获取支持向量：分别是决定分类器的支持向量，是最具有代表性的点

# 求解上边界和下边界
upper = vectors_[0]   # 取一个点
down = vectors_[-1]   # 再取一个点

upper_intercept = upper[1] - w*upper[0]   # 求上边界线截距 b = y-k*x
down_intercept = down[1] - w*down[0]  # 求下边界线截距 b = y-k*x

plt.scatter(X_train[:,0],X_train[:,1],c = y_train)
plt.scatter(vectors_[:,0],vectors_[:,1],s = 300,alpha=0.3)
plt.plot(x,w*x + b)  # 绘制边界线
plt.plot(x,w*x + upper_intercept)   #上边界的绘制
plt.plot(x,w*x + down_intercept)  #下边界的绘制


'''2.SVM分离坐标点案例-基于半径（rbf）'''
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

X_train = np.random.randn(200,2)   # 生成随机目标点
Y_train = np.logical_or(X_train[:,0]>0,X_train[:,1]>0)    # 逻辑算法：或
y_train = np.logical_xor(X_train[:,0]>0,X_train[:,1]>0)   # 逻辑算法：异或
plt.scatter(X_train[:,0],X_train[:,1],c = y_train)        # 将其分成两类,一三象限的数据是一类，二四象限的数据另一组

svc = SVC(kernel='rbf')    # 调用svm,kernel='rbf''表示基于半径
svc.fit(X_train,y_train)   # 进行训练  

# meshgrid扩展函数
xx,yy = np.meshgrid(np.linspace(-3,3,500),np.linspace(-3,3,500))  # 创造预测数据
# xy 平面中所有点
xy = np.c_[xx.ravel(),yy.ravel()]  # 得到平面全部的点
xx.shape
yy.shape
xy.shape

y_ = svc.decision_function(xy)   # xy中的点到分离超平面的距离？
d = y_.reshape(xx.shape)
# 进行绘制
plt.figure(figsize=(6,6))  # 将测试点到分离超平面的距离绘制成了一张图片
plt.imshow(d,extent=[-3,3,-3,3],cmap=plt.cm.PuOr_r)
plt.contourf(xx,yy,d)  # 绘制轮廓线、等高线，此圆圈上的测试点，到分离超平面的距离是相同的
plt.contour(xx,yy,d)
plt.scatter(X_train[:,0],X_train[:,1],c = y_train)   # 绘制点
plt.axis([-3,3,-3,3])  # 设定x轴与y轴的范围



'''案例3:使用多种核函数对iris数据集进行分类'''
import numpy as np
from sklearn.svm import SVC   # SVC继承自libsvm
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
iris = datasets.load_iris()   # 鸢尾花数据

X_train = iris.data
y_train = iris.target

X_train = X_train[:,[2,3]]   # 取其中两列
plt.scatter(X_train[:,0],X_train[:,1],c = y_train)

# 1.生成4种内核函数
linear_svc = SVC(kernel='linear')     # kernel='linear','linear'(线性)
poly_svc = SVC(kernel='poly')         # kernel='poly','poly'(多项式)
rbf_svc = SVC(kernel='rbf')           # kernel='rbf', 'rbf'(Radial Basis Function:基于半径函数),最常用效果也最好！
sigmoid_svc = SVC(kernel='sigmoid')   # kernel='sigmoid',

# 2.再来1种另外的SVC
# 这里是对 LinearSVC的介绍-该算法类似与SVC(kernel='linear') 
# Similar to SVC with parameter kernel='linear', 
# but implemented in terms of liblinear rather than libsvm
from sklearn.svm import LinearSVC
lsvc = LinearSVC()

# 3.将五种算法写成一个字典
estimators = {'linear_svc':linear_svc,'poly_svc':poly_svc,
              'rbf_svc':rbf_svc,'sigmoid_svc':sigmoid_svc,
             'lsvc':lsvc}

for i,key in enumerate(estimators):   # 看看这个与下面的区别
    print(i,key)
      
for i,key in estimators.items():   # 看看这个与下面的区别
    print(i,key)    

# 4.训练数据的找出最大与最小值   
xmin,xmax = X_train[:,0].min(),X_train[:,0].max()   # 获取训练数据的范围
ymin,ymax = X_train[:,1].min(),X_train[:,1].max()   # 获取训练数据的范围
xx,yy = np.meshgrid(np.linspace(xmin,xmax,700),np.linspace(ymin,ymax,300))
xy = np.c_[xx.ravel(),yy.ravel()]  # 交叉X轴和Y轴的点
 
# 5.进行训练---循环下训练---看看5种情况的差别！
plt.figure(figsize=(12,9))
for i,key in enumerate(estimators):
    esitmator = estimators[key]   
    esitmator.fit(X_train,y_train)  # 进行训练,首先是linear_svc.fit(X_train,y_train) 
    y_ = esitmator.predict(xy)  # 预测
    axes = plt.subplot(2,3,i+1)
    axes.pcolormesh(xx,yy,y_.reshape((300,700)),cmap = 'cool') 
    axes.scatter(X_train[:,0],X_train[:,1],c = y_train,cmap = 'rainbow')  
    axes.set_title(key)

# 6.关于meshgrid扩展函数
import numpy as np
nx,ny = (3,3)
x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)
xv,yv = np.meshgrid(x,y)
print(xv.ravel())
print(yv.ravel())












