# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:43:41 2018

@author: 肖
"""

#1.1 给定一个4维矩阵，如何得到最后两维的和？  
import numpy as np
n = np.random.randint(0,100,size = (2,2,3,3))  # 什么是最后2维的和，在这里表现为3*3的矩阵的和
n.shape
n.sum(axis = (2,3))  # axis = (2,3)表示编号为第二个和第三个(即3，3)（0开头）
n.sum(axis = (-1,-2))  # 最后数第一个和第二个(即3，3)

# 1.2 分布计算也可
nd1 = np.random.randint(0,10,size=(1,2,3,4))
nd2 = nd1.sum(axis=3)
nd2.sum(axis=2)

# 1.3矩阵的每一行的元素都减去该行的平均值?
n2 = np.random.randint(0,10,size = (4,5))
row_mean = n2.mean(axis = 1)  # 这里axis=1表示对(4,5)中的5求平均值；也即是行平均值
row_mean = row_mean.reshape((4,1))
print(n2 - row_mean)

# 2.如何根据第3列来对一个5*5矩阵排序!---思考方向-求出第三列排序的索引
nd = np.random.randint(0,100,size=(5,5))
nd[:,3]
np.sort(nd)  # 每行从大到小排序
np.sort(nd[:,3])  # 第三列进行排序
ind = np.argsort(nd[:,3])  # 拿出索引的排序
nd[ind]  # 得到结果

# 3.关于numpy的步长使用---n3[2:7:2,]
n3 = np.random.randint(0,10,size = (8,8))
n3[:2,]  # 表示输出编号第0行和第1行，所以总共有两行
n3[2:7:2,] # 2:7:2表示输出编号为第2行，第4行，第6行；2:7是始末，最后一个2是步长；

n3 = np.zeros((8,8),dtype=int)
n3[::2,]
n3[::2,1::2]=1
print(n3)

# 4.nupmy图形处理技术
import numpy as np
import matplotlib.pyplot as plt
fish = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/fish.png')
plt.imshow(fish)
print(fish.shape)  # (243, 326, 3)

fish2 = fish[::-1]  # 对高(243)进行颠倒
fish2.shape  # (243, 326, 3)
plt.imshow(fish2)

'''如果不理解参考下面内容:
a = np.random.randint(-100,100,size = (4,5))
a[::-1]
a[:,::-1]'''

fish3 = fish[::,::-1]  # 对长(326)进行颠倒
fish3.shape  # (243, 326, 3)
plt.imshow(fish3)

fish4 = fish[::,::,::-1]  # 对颜色(3)进行颠倒
fish4.shape  # (243, 326, 3)
plt.imshow(fish4)  # 得到颜色完全反过来的图片

fish5 = fish[::5,::5]  # 对图像元素进行抽取，这样得到的结果会非常模糊
fish5.shape  # (49, 66, 3)
plt.imshow(fish5)  

fish6 = fish.copy()  # .copy是直接复制！！！
fish6.shape  # (243, 326, 3)
fish6[80:120,80:110] = np.ones((40,30,3))  # 这样处理会使图片的部分被破坏
plt.imshow(fish6)

