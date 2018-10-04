# -*- coding: utf-8 -*-
"""
@author: 肖
"""

'''
1.K-means算法原理
聚类的概念：一种无监督的学习，事先不知道类别，自动将相似的对象归到同一个簇中。
K-Means算法是一种聚类分析（cluster analysis）的算法，其主要是来计算数据聚集的算法，
主要通过不断地取离种子点最近均值的算法。其原理仍是欧氏距离，过程如下：
随机在图中取K（这里K=2）个种子点。然后对图中的所有点求到这K个种子点的距离，假如点Pi离种子点Si最近，那么Pi属于Si点群。
接下来，我们要移动种子点到属于他的“点群”的中心。然后重复上面步骤，直到，种子点没有移动'''

'''2.K-means运用案例'''
import numpy as np
from sklearn.datasets import make_blobs  # 随机生成点
from sklearn.cluster import KMeans  # K-means:k均值聚类 cluster(一簇，一类)
import matplotlib.pyplot as plt

X_train,y_train = make_blobs(n_samples=150,centers=3,cluster_std=1)   # 生成随机点,150个点,3个中心点,标准方差为1
plt.scatter(X_train[:,0],X_train[:,1],c = y_train)   # 绘制原始图片
kmeans = KMeans(n_clusters=3)  # 调用k-means算法,n_clusters=3表示分成3组
kmeans.fit(X_train)            # 进行训练;无监督学习,没有目标,根据距离,自动分类
y_ = kmeans.predict(X_train)   # 预测结果
cluster_centers_ = kmeans.cluster_centers_     # 获取3个中心点的坐标！

plt.scatter(X_train[:,0],X_train[:,1],c = y_)       # 绘制训练后的分类图片,颜色不重要只看是否分类成功。
plt.scatter(cluster_centers_[:,0],cluster_centers_[:,1],s = 300,alpha = 0.7,c = [-2,1,2],
            cmap = plt.cm.winter_r)   # 画出三个中心点（种子点）的位置，s = 300是调整点的大小，alpha设置点的透明度，cmap调整图片显示格式


'''3.实战案例---中国足球几多愁'''
from mpl_toolkits.mplot3d import Axes3D   # 绘制3维图
import pandas as pd
from pandas import Series,DataFrame
ball = pd.read_csv('G:/python doc/spyder doc/AsiaFootball.txt',header=None)  # 50名以后均视作50名
ball.columns = ["国家","2006世界杯","2010世界杯","2007亚洲杯"]  # 重设列索引
kmeans = KMeans(3)   # 分成3组
kmeans.fit(ball[["2006世界杯","2010世界杯","2007亚洲杯"]])  # 进行数据的训练(与上例不同之处在于这里是一组3维数据)，注意这里需要2个中括号
y_ = kmeans.predict(ball[["2006世界杯","2010世界杯","2007亚洲杯"]])   # 进行预测,可以看到机器学习的结果

np.where(y_ == 1)     # numpy的两种值查找方式
np.argwhere(y_ == 0)  # numpy的两种值查找方式

# 将分到一类的国家分别打印出来
for i in range(3):
    index = np.argwhere(y_ == i)  
    for i, in index:     # 这里加逗号可以直接取出值,这里和元组很类似
        print(ball['国家'][i],end = ' ')
    print('\n')

# 绘制三维立体图形
plt.figure(figsize=(9,9))
axes3d = plt.subplot(projection = '3d')  # 绘制3维图
axes3d.scatter3D(ball['2006世界杯'],ball['2010世界杯'],ball['2007亚洲杯'],c = y_,cmap = 'rainbow')   # 分别表示x轴，y轴，z轴
cluster_centers_ = kmeans.cluster_centers_  # 接收3个中心点,当然这是三维数据
axes3d.scatter3D(cluster_centers_[:,0],cluster_centers_[:,1],cluster_centers_[:,2],
                 c = [-1,3,5],cmap = plt.cm.cool,s = 300,alpha = 0.5)    # 绘制3个聚类中心点



'''补充：print中逗号的使用---2个例子'''
import pandas as pd
from pandas import Series,DataFrame
print('apple',end = ' + ')

ball = pd.read_csv('G:/python doc/spyder doc/AsiaFootball.txt',header=None)  # 50名以后均视作50名
ball.columns = ["国家","2006世界杯","2010世界杯","2007亚洲杯"]  # 重设列索引
for i in range(6):
    print(ball['国家'][i],end = ' + ')










