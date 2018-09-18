#Talib Talib是python用于技术分析的包,包括：
#重叠研究（Overlap Studies）
#动能指标（Momentum Indicators）
#周期指标（Cycle Indicators）
#成交量指标（Volume Indicators）
#波动率指标（Volatility Indicators）
#数学操作（Math Operators）
#数学变换（Math Transform）
#统计功能（Statistic Functions）
#价格转换（Price Transform）
#形态识别（Pattern Recognition）



#计算均线(ta.MA)
import talib as ta
import pandas as pd
import talib.abstract as abstract
data = pd.read_excel('sz50.xlsx', sheetname='600036.XSHG', index_col='datetime')
#读取'numpy.ndarray'
print(data.head())
print(data.close.values) #data.close.values会返回一个numpy array格式的数列
print(type(data.close)) #验证上述
print(type(data.close.values)) #验证上述

#计算收盘价均线:
print(ta.MA(data.close.values, 5)) #用ta.MA计算5日均线
print(ta.MA(data.close.values, 5)[-5:]) #用ta.MA计算5日均线,打印最后五个
print(ta.abstract.MA(data, 5).tail()) #用ta.abstract计算5日均线,打印最后五个
#注意前面两者不同之处在于data.close.values为numpy array格式的数列,data为dataframe格式,会默认为输出第一列(收盘价)

#计算成交量的均线
data.volume=data.volume.astype('float64') #将volume列的格式改为float64  #！！！
print(ta.abstract.MA(data,5,price='volume').tail()) #price='volume'表示输出5日成交量均线
 










