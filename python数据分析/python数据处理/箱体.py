# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:36:59 2018
@author: 肖
"""
import numpy as np
import pandas as pd
from pandas import Series,DataFrame 

stock1 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600036.XSHG', index_col='datetime')
close = stock1.close

# 定义最大回撤率函数：
def MaxDrawdown(df):
    i = np.argmax((np.maximum.accumulate(df) - df) / np.maximum.accumulate(df))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(df[:i])  # 开始位置
    return ((df[j] - df[i]) / (df[j]))

# 定义最大收益率函数：
def MaxReturn(i):
    max_close = np.max(i)
    min_close = np.min(i)
    return ((max_close - min_close) / min_close)

# 计算并保存最大回撤率:
maxdrawdown = []
for i in range(1,len(close)):
    close_test = stock1.close[0:i]
    drawdown = MaxDrawdown(close_test)
    print(drawdown)
    maxdrawdown.append(drawdown)

# 画出最大回测率的变化:
import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
maxdrawdown = pd.Series(maxdrawdown)
plt.plot(maxdrawdown,c = '#0033ff')
plt.grid(color = 'g',linestyle = '--',linewidth = 1)  

# 计算并保存最大收益率:
maxreturn = []
for i in range(1,len(close)):
    close_test = stock1.close[0:i]
    return_rate = MaxReturn(close_test)
    print(return_rate)
    maxreturn.append(return_rate)


  
# 法1：用自定义的比例形成箱体
stock1 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600036.XSHG', index_col='datetime')
close = stock1.close
maxdrawdown = []
maxreturn = []
day = Series(data = np.random.randint(0,1,size = len(close)),index=stock1.index)   
for i in range (5,len(close)):
    close_test = close[0:i]
    drawdown = MaxDrawdown(close_test)
    return_rate = MaxReturn(close_test)
    maxdrawdown.append(drawdown)
    maxreturn.append(return_rate)
    if maxdrawdown[-1]>=0.2 or maxreturn[-1]>=0.2:
       day[i] = 1   # 标记1代表箱体的结束
       print('箱体结束!')
       maxdrawdown = []
       maxreturn = []
       close.drop(close.index[0:i],inplace=True)
    else:
        pass
  
             
# 法2:根据图形本身形成箱体
stock1 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600036.XSHG', index_col='datetime')
close = stock1.close
day = Series(data = np.random.randint(0,1,size = len(close)),index=stock1.index) 
close_test = close[0:60]  
for i in range (60,len(close)):
    max_1 = np.max(close_test)
    min_1 = np.min(close_test)    
    close_test = close[i:] 
    for j in range (i,len(close)):     
        if close[j]>max_1 or close[j]<min_1:
           day[j] = 1
           print('箱体结束!')
           i = i+j
        else:
            pass
            






