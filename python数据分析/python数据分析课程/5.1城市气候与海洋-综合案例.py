# -*- coding: utf-8 -*-
"""
@author: 肖
"""

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

'''1.数据准备'''
# 1.1以下城市均为海滨城市，距海洋距离有远有近:
citis = np.array(['asti','bologna','cesena','faenza','ferrara','mantova','milano','piacenza','ravenna','torino'])
asti = pd.read_csv('G:/python doc/spyder excel/Sea distance/asti_250715.csv')

# 1.2读取全部的数据
data = {}
for city in citis:   # 循环读取数据
    df1 = pd.read_csv('G:/python doc/spyder excel/Sea distance/%s_150715.csv'%(city))
    df2 = pd.read_csv('G:/python doc/spyder excel/Sea distance/%s_250715.csv'%(city))
    df3 = pd.read_csv('G:/python doc/spyder excel/Sea distance/%s_270615.csv'%(city))
  # 同一个城市的数据,pandas级联方法:
    df = pd.concat([df1,df2,df3],ignore_index=True) # ignore_index=True表示忽略行索引
  # 删除没用的列：
    df.drop(labels='Unnamed: 0',axis = 1,inplace=True)  
    data[city] = df

# 读取部分数据保存到列表中
data['milano']  
city_info = []
for city,df in data.items():  # city,df分别是data的一组键值对
    temp_max = df['temp'].max()   # 获得最大温度
    temp_min = df['temp'].min()   # 获得最小温度
    humidity_max = df['humidity'].max()     # 获得最大湿度
    humidity_min = df['humidity'].min()     # 获得最小湿度
    humidity_mean = df['humidity'].mean()   # 获得平均湿度
    dist = df['dist'][0]                    # 距离取一个就行,这里取第一个数据
    l = [temp_max,temp_min,humidity_max,humidity_min,humidity_mean,dist,city]   
    city_info.append(l)

# 1.3将列表city_info转换成dataframe  
city_info_df = DataFrame(city_info,columns=['temp_max','temp_min','humidity_max','humidity_min','humidity_mean','dist','city'])   


'''2.数据分析与普通线性回归'''
# 2.1画点图---展示横纵坐标，横坐标表示距离，纵坐标表示对应的最大温度
plt.scatter(city_info_df['dist'],city_info_df['temp_max'])  
  
# 2.2使用机器学习，将温度和与海洋远近的关系线，绘制出来，线性关系f(x) = w*x + b
# 首先使用numpy：把列表转换为numpy数组，用于后续计算。分别以100公里和150公里为分界点，划分为离海近和离海远的两组数据。 
cond = city_info_df['dist'] < 150  # 根据条件去DataFrame中获取数据
city_near = city_info_df[cond]
cond = city_info_df['dist']> 100
city_far = city_info_df[cond]  

from sklearn.linear_model import LinearRegression  # 导入机器学习算法包，线性回归
lrg = LinearRegression()  # 调用线性回归的算法
# 进行普通线性回归训练:注意下,X=、y=的这种写法也是可以
lrg.fit(X=city_near[['dist']],y=city_near['temp_max'])  # 注意训练数据X必须是二维

# a.获取近海的回归曲线 y=k*x+b
# 获取斜率,系数,weight：
w_ = lrg.coef_[0]
# 获取斜率,偏差,bias：
b_ = lrg.intercept_

# b.获取远海的回归曲线 y=k*x+b
lrg.fit(city_far[['dist']],city_far['temp_max'])
w2_ = lrg.coef_[0]    # 斜率
b2_ = lrg.intercept_  # 斜率
 
# 2.3画出回归结果 
plt.scatter(city_info_df['dist'],city_info_df['temp_max'])  # 先画出原来的点图 
x = np.linspace(0,150,100)
y = w_*x + b_
plt.plot(x,y,color='green',label = 'near sea')
x2 = np.linspace(120,350,100)
y = w2_*x2 + b2_   # 第二条曲线
plt.plot(x2,y,color='red',label='far sea')
plt.legend()  # 必须添加这行分别添加才能生效

# 3.1 查看海洋距离与最低温度的关系
plt.scatter(city_info_df['dist'],city_info_df['temp_min'])
# 3.2 查看海洋距离与最低湿度的关系
plt.scatter(city_info_df['dist'],city_info_df['humidity_min'])
# 3.3 查看海洋距离与最高湿度的关系
plt.scatter(city_info_df['dist'],city_info_df['humidity_max'])
# 3.4 查看海洋距离与平均湿度的关系
plt.scatter(city_info_df['dist'],city_info_df['humidity_mean'])

# 3.5 查看风向与风速的关系---这里使用了柱状图,极坐标图
wind_deg = data['milano']['wind_deg']      # 获取风向
wind_speed = data['milano']['wind_speed']  # 获取风速
# np.histogram函数将数据平分为8份
wind_count,wind_range = np.histogram(wind_deg,bins=8,range = [0,360])

# a.画柱状图
plt.bar(wind_range[:-1],wind_count,width = 22.5)  # 画柱状图,或者是条形图

# b.画极坐标图
plt.figure(figsize=(9,9))
plt.axes(polar = True,facecolor = 'green')
plt.bar(np.arange(0,2*np.pi,np.pi/4),wind_count,color = np.random.rand(8,3))

# c.大于45度且小于90度的风速情况统计
cond = (wind_deg < 90)&(wind_deg >=45)
wind_speed[cond].describe()

















