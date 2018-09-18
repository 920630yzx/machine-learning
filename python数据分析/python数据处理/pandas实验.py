# 1.计算n日百分比变化,pct_change()
import pandas as pd
stock1 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600036.XSHG', index_col='datetime')
one_day_change = stock1.close.pct_change()
five_day_change = stock1.close.pct_change(5) # 计算5日变化率
ten_day_change = stock1.close.pct_change(10) # 计算10日变化率
import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
plt.plot(one_day_change.iloc[-50:], label='one') # 这里打印的是最后50行
plt.plot(five_day_change.iloc[-50:], label='five')
plt.plot(ten_day_change.iloc[-50:], label='ten')
plt.legend()
plt.show()



# 2.计算dataframe协方差cov(),相关系数corr()
stock2 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600050.XSHG', index_col='datetime')
stock3 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='601318.XSHG', index_col='datetime')
five_day_df = pd.concat([stock1.close.pct_change(5),stock2.close.pct_change(5),stock3.close.pct_change(5)],keys=['stock1','stock2','stock3'],axis=1)
#通过concat将其组合成DataFrame,keys分别代表每列的列名，axis=0会组合成1列，因此axis必须=1，这样才会有三列
print(five_day_df.tail()) # tail()打印最后5行
print(five_day_df.cov()) # 计算收益率变化的协方差
correlation = five_day_df.corr()  # 计算收益率变化的相关系数
print(correlation)
spearman = five_day_df.corr(method='spearman') # 计算排序相关系数,另一种相关性的计算方法
print(spearman)



# 3.排序(简单)
print(five_day_df.tail())
print(five_day_df.tail().rank())  # 正向纵序（由小到大）
print(five_day_df.rank(axis=1, ascending=False).tail()) # 横向倒序（由大到小）



# 4.滚动计算rolling()
rolling = stock1.close.rolling(window=20) # 计算20日均线,rolling = stock1.close.rolling(20)好像完全一样，也可行
plt.figure(figsize=(15,7))
plt.plot(rolling.mean())
plt.plot(stock1.close)
plt.show()


# 5.调用自定义函数apply！   (这里列举10日均线波动幅度的计算方法)
import numpy as np
def cal_range(array):
    return array.max() - array.min()

print(stock1.rolling(10))
print(stock1.rolling(10).apply(cal_range))
print(stock1.rolling(10).apply(cal_range).tail()) # 这里计算10日均线波动幅度



# 6.50天滚动的五日收益的协方差？？？
cov_50 = five_day_df.rolling(50).cov()  # 50天滚动的五日收益协方差(50天滚动的协方差)
print(cov_50.tail(6))  # 打印后6个
# 50天滚动的五日收益相关性
corr_50 = five_day_df.rolling(50,).corr() # 50天滚动的五日收益相关性
print(corr_50.tail(6))

plt.figure(figsize=(15,7))
plt.plot(corr_50.unstack()['stock1','stock3'])   # A = corr_50.unstack()
#unstack()有dataframe的作用,利用['stock1','stock3']找出'stock1'与'stock3'相关系数
plt.show()



# 7.聚合aggregate(),主要用于处理数据运算(诸如统计平均值,求和等),并返回计算后的运算结果。!!!
# 7.1对dataframe运用agg(agg=aggregate)
sum = five_day_df.agg(np.sum)
sum_agg = five_day_df.rolling(10).agg(np.sum)  # 对多只股票收益率进行滚动求和
sum_agg_2 = five_day_df.rolling(10).aggregate(np.sum)  # 对多只股票收益率进行滚动求和
# 7.2对Series运用agg(agg=aggregate)
five_day_df.stock1  # 即five_day_df.iloc[:,0],也即five_day_df第一列;并且可以看出agg=aggregate
multi_agg = five_day_df.stock1.rolling(10).agg([np.sum, np.mean, np.std])
multi_agg_2 = five_day_df.stock1.rolling(10).aggregate([np.sum, np.mean, np.std])  
# 7.3对dataframe的各列进行多种不同的计算,分别计算滚动的和,均值,方差
multi_algo = five_day_df.rolling(10).agg({'stock1' : np.sum,
       'stock2' : np.std,
       'stock3' : np.mean}) # 针对不要数据用不同算法,与前者有许多相似之处
print(stock1.resample('W').tail())  # resample('W')抽样方法计算,由于没写函数方法,会默认对每一列调用求均值的方法
print(stock1.resample('W').agg({'high':'max','low':'min', 'close':'last', 'open':'first','volume':'mean'}).tail())
# resample成周线后用agg获取所需要的数据，用这种方式画出周K线;'high':'max',也即high列最高价;输出的顺序也做了自动的调整;mean是求均值,sum是求和



# 8.expanding()  expanding不会返回NaN,功能与rolling相似；cumsum()分步求和
sn = pd.Series([1, 2, np.nan, 3, np.nan, 4])
print(sn.expanding().sum()) # 求和sn.sum(),sn.expanding().sum()分步求和
print(sn.cumsum()) # 与前者一样分步求和，不过会有NAN
print(sn.cumsum().fillna(method='ffill')) #向前填充
sn = pd.Series([1, 2, 2.5, 3, 3.5, 4, np.nan, 5, 6])
print(sn.rolling(2).sum()) # 求和,rolling(2)会出现NAN



# 9.SMA的递归计算,先做了解
def SMA(A,n,m):
    # 设置alpha的比例
    alpha = m/n
    #通过ewm计算递归函数
    return A.ewm(alpha=alpha, adjust=False).mean()
print(SMA(stock1.close,7,2).tail())