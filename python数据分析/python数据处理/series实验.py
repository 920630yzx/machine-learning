# Series Series是pandas系列里的一维数组，它可以包含任何数据类型的标签。
# 我们主要使用它们来处理时间序列数据。



# 1.创建Series 并修改名称、修改索引
import pandas as pd
import numpy as np
from datetime import datetime
s=pd.Series([1,2,np.nan,4,5])
print(s)
print(s.name)
s.name='price'  # 给Series添加名称
print(s)
print(s.name)
new_index = pd.date_range('20160101',periods=len(s),freq='D')  # periods表示长度
s.index = new_index  # 修改索引
print(s)



# 2.1访问series元素  iloc[],loc[]
# 我们使用iloc[]来访问元素的整数索引和我们使用loc[]来访问指数序列的。
print("First element of the series: ", s.iloc[0])
print("Last element of the series: ", s.iloc[len(s)-1])
print(s.iloc[0:5:2])  # 批量读取 0:5:2表示0、5为初始值,最终值,2为步长

print(s.loc['20160101'])  # 读取索引
print(s.loc['20160101':'20160103'])   # 批量读取索引



# 2.2布尔索引  除了上述访问方法,您可以使用布尔过滤序列数组。比较序列与标准是否一致。
# 当与您设定的任何条件相比,这次你返回另一个系列中,回填满了布尔值。
print(s<3)  # 若s<3则返回为true
print(s.loc[s<3])  # 返回小于3的行
print(s.loc[(s < 3) & (s > 1)])

s.head(3)  # s的头三个
s.head(-1) # s的第一个至倒数第二个



# 3.通过excel获取数据,进行resample()、data_range()、reindex()。
import pandas as pd
data = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx', sheetname=0, index_col='datetime')
print(data.index)
Series = data.close  # Dataframe的读取方式
Series.head()

# 3.1用resample给每个月的最后一天抽样：
monthly_prices = Series.resample('M').last()  # resample('M')表示按月取值,last()表示取每月最后一天的收盘价
print(monthly_prices.head(10))
monthly_prices_first = Series.resample('M').first()  # first()表示求每月月初的值
print(monthly_prices_first.head(10)) 
monthly_prices_med = Series.resample('M').median()   # median()表示求每月中间取值
print(monthly_prices_med.head(10))
monthly_prices_mean = Series.resample('M').mean()    # median()表示求每月均值
print(monthly_prices_med.head(10))
monthly_prices_sum = Series.resample('M').sum()    # median()表示求每月的和
print(monthly_prices_med.head(10))

# 3.2 series计算 .apply()调用自定义的函数,Series求和与均值
print(Series.loc[datetime(2017,1,1):datetime(2017,1,10)])
sum(Series.loc[datetime(2017,1,1):datetime(2017,1,10)])
sum(Series.loc[datetime(2017,1,1):datetime(2017,1,31)])/len(Series.loc[datetime(2017,1,1):datetime(2017,1,31)])

# 修改Series的索引，并求其Series的和
# I = list(range(len(Series)))
# Series.index = I
# print(Series.loc[0:4])
# sum(Series.loc[0:4])

# apply()调用自定义函数:
def custom_resampler(array_like):
    """ Returns the first value of the period """
    return array_like[0]  # 返回第一个
first_of_month_prices = Series.resample('M').apply(custom_resampler)  # 这个结果当然与monthly_prices_first完全相同
first_of_month_prices.head(5)

# 3.3series滚动计算  rolling
rolling_mean = Series.rolling(window=40).mean()  # window=40.mean()表示求40日均线,rolling表示滚动计算
rolling_std = Series.rolling(window=40).std()  # 求40日滚动方差



# 4.series的填充及缺失数据处理fillna(method='ffill'),resample('D');
# resample()除了有求一定时间段的作用还会自动补齐不足的时间段
from datetime import datetime
data_s= Series.loc[datetime(2017,1,1):datetime(2017,1,10)]  # 读取时间为datetime(2017,1,1):datetime(2017,1,10)的行
data_r=data_s.resample('D').mean()  # resample('D')会自动补齐没有的天数,空值自动为NAN；mean()为求日均值，如果不写视为不全
print(data_r.head(10))

print(data_r.fillna(method='ffill'))   # s的前填充,这不会改变原值
print(data_r.fillna(method='bfill'))   # s的后填充,这不会改变原值
data_r.head(5).fillna(method='ffill')  # s的前填充,但是只读取前5个
data_r.head(5).fillna(method='bfill')  # s的后填充,但是只读取前5个
data_r.dropna()   # s的填充,去掉NAN属性,这也不会改变原值



# 5.1Series做图
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 7))
plt.plot(Series)  # 画出Series列
# Series.plot()  # 画出Series列,区别在于Series.plot()法画出的会填充的更饱满
plt.title("Stock Prices")
plt.ylabel("Price")
plt.xlabel("Date")
plt.show()

# 5.2 DataFrame与Series的计算
print("Mean: ", data.mean())  # 求data每列均值
print("Standard deviation: ", data.std())  # 求data每列的方差
print("Summary Statistics", data.describe()) # data描述性统计
add_returns = Series.diff()[1:]  # 求Series的差值变化，[1:]表示结果省略第一行
mult_returns = Series.pct_change()[1:]  # 求Series的每行增长率

# 5.3 作图1
plt.figure(figsize=(15, 7))
plt.title("returns of Stock")
plt.xlabel("Date")
plt.ylabel("Percent Returns")
mult_returns.plot()  # 画出mult_returns
plt.show()

# 5.3 作图2
rolling_mean = Series.rolling(window=40).mean()   # window=40.mean()表示求40日均线,rolling表示滚动计算
rolling_mean.name = "40day rolling mean"   # 给rolling_mean命名
plt.figure(figsize=(15, 7))
Series.plot()   # 画出Series
rolling_mean.plot()   # 画出rolling_mean,这样一张图就画了两条线
plt.title("Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# 5.3 作图3
rolling_std = Series.rolling(window=40).std()  # 求40日滚动方差
rolling_std.name = "40day rolling volatility" 
plt.figure(figsize=(15, 7))
rolling_std.plot()  # 画出rolling_std
plt.title(rolling_std.name)
plt.xlabel("Date")
plt.ylabel("Standard Deviation")
plt.show()


















