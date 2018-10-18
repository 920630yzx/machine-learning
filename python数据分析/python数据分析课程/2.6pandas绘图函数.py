# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 16:12:50 2018
@author: 肖
"""

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt

'''1.线性图---plot函数默认绘制线形图'''
# 1.1 画1条线性图
s = Series(np.random.randint(0,100,size = 10),index = list('abcdefghij'))
s.plot()

# 1.2 画1条线性图
s = Series([80,81,85,90,80],name = "数学", index = list("abcde"))
s.plot()

# 1.3 画n条线性图
df = DataFrame(np.random.randint(0,50,size = (10,4)),columns=list('ABCD'))
df.plot()



'''2.柱状图---绘制线形图柱状图'''
# 柱状图1： .plot(kind = 'barh')
df = DataFrame(np.random.randint(0,50,size = (10,4)),columns=list('ABCD'))
df.plot(kind = 'barh')

# 柱状图2：.plot(kind='bar')
df1 = DataFrame({'day':['Fri','Stat','Sun','Thur'],'1':[1,2,0,1],'2':[16,53,39,48],'3':[1,18,15,4],'4':[1,13,18,5],'5':[0,1,3,1],'6':[0,0,1,3]},
                index=[0,1,2,5],
                columns=['day','1','2','3','4','5','6'])
df1.set_index('day',inplace=True)  # s！！！et_index函数修改索引
df1.plot(kind='bar')
df.plot(kind = "bar", colormap = "rainbow")  # colormap = "rainbow"绘制成彩虹颜色

# 柱状图3： --- 横纵坐标互换后的图
df1.stack()  # 先一维化
df1.stack().unstack(level = 0)  # 第一级索引转化为列名称
df1.stack().unstack(level = 0).plot(kind = 'bar')



'''3.绘制直方图---hist绘制直方图,它是一种特殊的柱状图,该图用来表示密度'''
nd = np.random.randint(0,100,size = 100)
s = Series(nd)
s.hist(bins = 300)  # bins = 500表示线条的粗细，越小越粗



'''4.绘制随机数百分比密度图---.plot(kind='kde')'''
s.plot(kind='kde')



'''5.直方图,和密度图绘制到一个图形中（有点无聊，不是真正意义上的一张图）'''
n1 = np.random.normal(loc = 0,scale=1,size = 100)  # 随机正太分布，以0为均值，1为方差
n2 = np.random.normal(loc = 10,scale = 2,size = 100)
nd = np.concatenate([n1,n2])  # numpy的级联，注意与pandas的级联concat区分开
s = Series(nd)

s.hist(bins = 100)  # 绘制直方图
s.hist(normed = True,bins = 100)  # 绘制直方图，normed = True表示将元素标准化
s.plot(kind = 'kde',style = 'red')  # kind = 'kde' 密度图



'''6.绘制散布图---plot(...,kind='scatter')'''
df = DataFrame(np.random.randint(0,100,size = (50,4)),columns=list('ABCD'))
print(df)
df.plot(x='A',y='A',kind='scatter')  # df中A列与A列的关系比较
df.plot(x='A',y='B',kind='scatter')  # df中A列与B列的关系比较
df.plot(x='A',y='C',kind='scatter')  # df中A列与C列的关系比较

'''6.1绘制散布图矩阵---pd.plotting.scatter_matrix'''
pd.plotting.scatter_matrix(df,diagonal='kde')   # 散布图矩阵，当有多个点时，两两点的关系，diagonal='kde'表示绘制密度图
pd.plotting.scatter_matrix(df,diagonal='scatter')    # 散布图矩阵，当有多个点时，两两点的关系，diagonal='scatter'表示绘制散布图
















