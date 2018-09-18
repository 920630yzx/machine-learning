# -*- coding: utf-8 -*-
"""
@author: 肖
"""
'''8. 副本'''
# 8.1
import numpy as np
n = np.array([1,2,3])
n[2] = 10  # 第三个元素变为10
print(n)

n = n+1  # 每个元素加上1
print(n)

# 8.2创建副本
n2 = n.copy()  # 创建副本
n2[0] = 1024  # 修改副本，但是原来的n是不会发生变化的
print(n2)
print(n)



'''9.ndarray的聚合操作'''
# 9.1 axis = (0,1)
import matplotlib.pyplot as plt  # 导入画图
cat = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/cat.jpg') 
cat.shape  # (456, 730, 3)
cat.max()  # 255
cat.min()  # 0
cat.max(axis = 0)
cat.max(axis = 0).shape  # (730, 3)
cat.max(axis = 1)
cat.max(axis = 1).shape  # (456, 3)
cat_2 = cat.max(axis = 0)
cat_2.shape  #  (730, 3)
cat_3 = cat.max(axis = 1)
cat_3.shape  #  (456, 3)
cat_4 = cat.max(axis = (0,1))
cat_4.shape  #  (3,)

# 9.2 reshape的其他用法
cat2 = cat.reshape((-1,3))  # -1就代表前两者自动相乘了，是一种简便写法
cat2.shape  # (332880, 3)
cat3 = cat.reshape((456*730,3))  # 结果与前面完全一样
cat3.shape  # (332880, 3)
cat3.max(axis = 0)

# 9.3 其他聚和操作
A = np.array([1,0,0])
np.any(A)  # 有一个为true则返回true
np.all(A)  # 有一个为false则返回false
'''其他聚会操作
Function Name    NaN-safe Version                Description
np.sum           np.nansum                  Compute sum of elements
np.prod          np.nanprod                 Compute product of elements
np.mean          np.nanmean                 Compute mean of elements
np.std           np.nanstd                  Compute standard deviation
np.var           np.nanvar                  Compute variance
np.min           np.nanmin                  Find minimum value
np.max           np.nanmax                  Find maximum value
np.argmin        np.nanargmin               Find index of minimum value
np.argmax        np.nanargmax               Find index of maximum value
np.median        np.nanmedian               Compute median of elements
np.percentile    np.nanpercentile           Compute rank-based statistics of elements
np.any           N/A                        Evaluate whether any elements are true
np.all           N/A                        Evaluate whether all elements are true
np.power 幂运算'''

# 9.4 nan的聚和操作
nd1 = np.array([12,13,np.nan,14])
np.sum(nd1)
np.nansum(nd1)  # 忽视nan进行求和操作,结果为39

'''10.numpy计算'''
# 10.1矩阵相乘及转置
A=np.array([[1,2],
            [4,5],
            [7,8]])
B=np.array([[0,1,2],[3,4,5]])
C=np.dot(A,B)  # np.dot求矩阵之积
print(C)   # 打印矩阵
print(C.T) # 输出矩阵C的转置矩阵

# 10.2矩阵计算
n = np.random.randint(0,20,size = (4,5))
n2 = np.random.randint(0,5,size = (4,5))
n*2   # 每个元素乘以2
n*n2  # 每个元素对应相乘
n3 = np.random.randint(0,5,size = (5))
n*n3  # 每个元素对应相乘

# 10.3广播机制:相当于人人有份
# ndarray广播机制的两条规则  规则一：为缺失的维度补1；规则二：假定缺失元素用已有值填充
m = np.ones((2,3))
a = np.arange(3)
print(m,a)
print(m+a)
print(m*a)
print(m/a)
print(m+1)
print(m*3)
print(m+np.array([1,2,3]))
print(m+np.array([[1],[2]]))

# 10.4数组基本运算
import pandas as pd
import numpy as np
e = np.array([20,30,40,50])
f = np.arange(4)
print(e-f) #求差
print(e*2) #求积
g = np.linspace(0,np.pi,3) # 返回输入数量的等比间隔，linspace(start, stop, num=50)
print(g.sum()) #求和
print(g.cumsum())  # 累计求和
print(np.log(g))  # 求对数,对数组中每一个元素都求其对数
print(np.exp(g))  # 求指数,对数组中每一个元素都求其指数
print(np.sqrt(g)) # 求开方,对数组中每一个元素都求其开发
C=np.dot(A,B)  #求矩阵之积,A、B之前已定义;#求矩阵之积

# 10.5求逆矩阵np.linalg.inv()
X=np.array([[2,1],[1,-2]])
X_=np.linalg.inv(X)  # 求矩阵X的逆矩阵
print(X)
print(X_)
print(np.dot(X,X_))

# 10.6 求矩阵的秩
np.linalg.matrix_rank(X) 

# 10.7求矩阵的解线代  题目：
# 2x+y=1
# x-2y=1 则：
b1 = np.array([[1],[1]])  # 线代中的b值
X = np.array([[2,1],[1,-2]]) # 线代中的A值
X_= np.linalg.inv(X)
z = np.dot(X_,b1)  # 求矩阵解线代
print(z)


'''11.排序'''
# 11.1 冒泡排序法
n = np.random.randint(0,100,size = 10)
def bubble(nd):
    for i in range(nd.size):  # nd.size是计算总长度
        for j in range(i,nd.size):
            if nd[i] > nd[j]:
                nd[i],nd[j] = nd[j],nd[i]
    return nd
bubble(n)

# 其他排序法
def argindex(nd):
    for i in range(nd.size):
        index = np.argmin(nd[i:]) + i   #argmin 可以获取数据的最小值的索引
        nd[i],nd[index] = nd[index],nd[i]
    return nd
n = np.random.randint(0,100,size = 20)
np.argmin(n)
argindex(n)

# 11.2 快速排序
# np.sort()与ndarray.sort()都可以，但有区别：np.sort()不改变输入；
# ndarray.sort()本地处理，不占用空间，但改变输入(不常用)
A = np.random.randint(0,100,size=(2,3))
np.sort(A)  # 不改变A本身，升序排列
A = np.sort(A)  # 这样就可以改变A本身
np.sort(A,axis=0)  # 不改变A本身，列升序排列
np.sort(A,axis=1)  # 不改变A本身，行升序排列

A = np.random.randint(0,100,size=(10))
np.sort(A)[::-1]  # 最后加上一个'-1'这样就可以得到降序！！！
np.sort(A)[5:2:-1]  # 倒序的切片

# 11.3部分排序  np.partition(a,k)
# 有的时候我们不是对全部数据感兴趣，我们可能只对最小或最大的一部分感兴趣。
# 当k为正时，我们想要得到最小的k个数；当k为负时，我们想要得到最大的k个数
A = np.random.randint(0,100,size = 10)
np.partition(A,3)   # 前3个数为最小的数，但不会进行排序
np.partition(A,-3)  # 后3个数为最大的数，但不会进行排序

# 10.4 argwhere判断元素值（将满足条件的数的位置返回）
A = np.random.randint(-10,10,size=(10))
index = np.argwhere(A>0)  # 将元素值大于0的数值的位置返回
A[index] # 返回具体的值












