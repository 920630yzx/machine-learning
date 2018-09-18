import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt

# 1.1画二次函数图
x = np.linspace(0, 5, 10) # 0到5均分为10份
y = x ** 2
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.plot(x, y, 'r')  #'r'表示颜色,x, y分别表示自变量与因变量
axes.set_xlabel('x')  # x轴标签 
axes.set_ylabel('y')  # y轴标签 
axes.set_title('title');  # 标题
plt.plot(x, y, 'r')

# 1.2画二次函数图plt.plot也可画出,当然axes可以不要了
x = np.linspace(0, 5, 10) # 0到5均分为10份
y = x ** 2
fig = plt.figure()
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
plt.plot(x, y, 'r')  #'r'表示颜色,x, y分别表示自变量与因变量
plt.xlabel('x')  # x轴标签 
plt.ylabel('y')  # y轴标签 
plt.title('title');  # 标题



# 2.画子图   axes.plot画法优点在于可以画子图
fig = plt.figure() # 新开画布
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes 内置的图表;前面两个数字表示子表的位置,后面两个数字表示字表的宽和高
# 主图画二次函数
axes1.plot(x, y, 'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')
# 子图里画出该函数的反函数
axes2.plot(y, x, 'g') #画出反函数,'g'表示颜色为绿色
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('insert title');



# 3.1画Serise直方图 (plt.hist) (收盘价直方图):
stock1 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600036.XSHG', index_col='datetime')  # index_col会保留索引值;换index试试?结果..
stock1.head()
plt.figure(figsize=(15, 7)) # 设置图表的大小
plt.hist(stock1.close, bins=20) # plt.hist表示画柱形图(直方图),bins=20代表柱的数量

plt.xlabel('Price')
plt.ylabel('Number of Days Observed')
plt.title('Frequency Distribution of 000001 Prices, 2016')
plt.show() #可以看到收盘价出现的频率

# 3.2画Serise的回报率直方图：
R1 = stock1.close.pct_change()[1:] #pct_change()表示每日收益率,[1:]表示从第二行开始
plt.figure(figsize=(15, 7))
plt.hist(R1, bins=20)
plt.xlabel('Return')
plt.ylabel('Number of Days Observed')
plt.title('Frequency Distribution of 000001 Returns, 2016')
plt.show()

# 3.3画Serise的累积直方图;cumulative=True  (累积直方图有点类似于分布函数):
plt.figure(figsize=(15, 7))
plt.hist(R1, bins=20, cumulative=True)
plt.xlabel('Return')
plt.ylabel('Number of Days Observed')
plt.title('Cumulative Distribution of 600036 Returns, 2016')
plt.show()



# 4.1画两个Serise的散点图(scatter):
stock2 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='601318.XSHG', index_col='datetime') #增加一支股票
plt.figure(figsize=(15, 7))
plt.scatter(stock1.close, stock2.close, c = ['g','r'], s= 20) #s=20表示点的大小,c = ['g','r']分别表示颜色
plt.xlabel('600036')
plt.ylabel('601318')
plt.title('Daily Prices in 2017')
plt.show()

# 4.2画两个Serise的回报率散点图(scatter)：
R2 = stock2.close.pct_change()[1:]
plt.figure(figsize=(15, 7))
plt.scatter(R1, R2,c=['c','r'],s=20)
plt.xlabel('000001')
plt.ylabel('000005')
plt.title('Daily Returns in 2016')
plt.show()



# 5.1画两个Serise的线图      (线图直接plt.plot即可)
plt.figure(figsize=(15, 7))
plt.plot(stock1.close,c='r')  # c='r'表示红色，若不写也会有默认的不同种颜色
plt.plot(stock2.close,c='b')
plt.ylabel('Price')
plt.legend(['600036','601318']) # 给线条命名
plt.show()

# 5.2画两个Serise的回报率线图(plot)   plt.hlines在图中画虚线   alpha决定线条颜色深浅
plt.figure(figsize=(15, 7))
plt.plot(R1)
plt.hlines(0,R1.index[0],R1.index[-1],linestyle='dashed',alpha=0.3)
#画虚线,0,R1.index[0],R1.index[-1]表示在y=0的位置画长度为R1.index[0],R1.index[-1]的虚线
#linestyle='dashed'表示是画的虚线,alpha=0.3表示虚线颜色的深浅
plt.ylabel('Return')
plt.title('600036 Returns')
plt.show()

# 5.3subplot方法画两张图(两图合并) 
plt.figure(figsize=(15, 7))
plt.subplot(2,1,1)  # (2,1,1)表示画2*1矩阵的图，最后的1表示先画第一张图
plt.plot(stock1.close)
plt.subplot(2,1,2)
plt.plot(stock2.close)
plt.show()



# 6.1画K线   
# dataframe重新排列每一列的顺序reindex_axis()  fig,(ax,ax1)是同时画两张图的写法，两张图名字分别为ax,ax1
# sharex=True表示两张图共享x轴   ax.grid(True)表示加上网格线   调整图片比例fig.subplots_adjust(bottom=0.5)
from matplotlib.pylab import date2num 
stock1['time'] = list(map(date2num,stock1.index))  # 末尾加一列'time'，时间转换为数字形式
stock1.head()  # 可以看到变化
candle = stock1.reindex_axis(['time','open','high','low','close'],1).values  # 'time','open','high','low','close'表示取五个轴，并依次排序
#不要.values是dataframe格式，有.values则是float64格式，reindex_axis有重新排序的作用

import matplotlib.finance as mpf
fig,(ax,ax1)=plt.subplots(2,1,sharex=True,figsize=(15,25))
# fig,(ax,ax1)是同时画两张图的写法，两张图名字分别为ax,ax1;2,1表示两张图;sharex=True,表示x轴是共享的;figsize=(15,25)表示图片的大小
fig.subplots_adjust(bottom=0.5)  # 调整图片的比例
ax.grid(True)  # 第一张图加上网格线
mpf.candlestick_ohlc(ax,candle,width=0.6,colorup='r',colordown='g',alpha=1.0)  # ohlc表示以开盘价,最高价,最低价,收盘价作为顺序,ax表示画在ax这张图上
#candle为数据(必须为float型数据),colorup='r',colordown='g'分别代表上涨颜色与下跌颜色,alpha表示颜色深浅
ax1.bar(stock1.index,stock1.volume)  # bar函数,stock1.index,stock1.volume时间索引及成交量
ax.xaxis_date()  # 转为统一的时间格式
plt.show()




