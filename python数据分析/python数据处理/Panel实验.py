# 1.excel的循环读取为字典,字典转Panel
from datetime import datetime
import pandas as pd
symbol=['600036.XSHG','600050.XSHG','601318.XSHG']  # 定义列表
data_dict = {}  # 定义字典
for s in symbol:
    data =  pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname=s, index_col='datetime') #循环读取
    data_dict[s] = data.loc['2017-03-21':'2017-05-10']  # 只取'2017-03-21':'2017-05-10'时间段
PN = pd.Panel(data_dict) #将字典转换成Panel
print(PN) # 3 (items) x 34 (major_axis) x 5 (minor_axis)
# items表示股票名，major表示时间(主索引)，minor_axis表示股票高开低收成交量(附索引)

MI = PN.to_frame()  # Panel特有读取方式，to_frame()表示详细数据
round(MI.head(),1)  # Panel特有读取方式，默认状态下按照主索引读取，round(,1)表示保留一位小数



# 2.1Panel分类及抽样  猜想：数字越大优先级越高，第一个数字代表(主索引)，第二个数字代表(附索引)，第三个数字是items？
MI_tp = PN.transpose(2,1,0).to_frame()    # 先时间后股票分类
MI_tp_1 = PN.transpose(1,2,0).to_frame()  # 按照高开低收分类
MI_tp_2 = PN.transpose(2,0,1).to_frame()  # 先股票后时间分类
MI_tp_3 = PN.transpose(0,1,2).to_frame()  # 先股票后时间分类
MI_tp_4 = PN.transpose(0,2,1).to_frame()  # 先股票后时间分类
MI_tp_5 = PN.transpose(1,0,2).to_frame()  # 先股票后时间分类

# 2.2修改Panel的Items名称
PN_rename = PN.rename(items={'600036.XSHG':'ZSYH','600050.XSHG':'ZGLT','601318.XSHG':'ZGPA'})
#'600036.XSHG':'ZSYH'表示将600036.XSHG名称改为ZSYH,修改的Items axis列
print(PN_rename)



# 2.3Panel的抽样取每周星期一数据  transpose(2,1,0).resample('W-MON',axis=1)
PN_RE = PN.transpose(2,1,0).resample('W-MON',axis=1).last()   # .last()的作用好像是保留小数位,不是取打印最后5行
# 'W-MON'W表示每周,MON表示以周一进行抽样；axis=1即以major_axis轴进行抽样
print(PN_RE)
print(PN_RE.to_frame()) # 访问详细数据
print(PN_RE.to_frame().head())



# 3.Panel数据访问！！！
print(PN_RE['close'].head())  # 访问items(股票)的数据,即所有股票收盘价打印出来
print(PN_RE.major_xs('2017-04-10'))  # 访问major(某一天)的数据,major_xs即major_axis
print(PN_RE.minor_xs('600036.XSHG').head()) # 访问minor(某只股票)的数据,minor_xs即minor_axis

# 访问loc使用名称索引   分别表示item，major_axis,minor_axis
print(PN_RE.loc[:,'2017-03-21':'2017-04-10',:].to_frame())
# 访问loc使用名称索引
print(PN_RE.iloc[2:,2:,2:])
# 访问ix使用名称索引 ！！！
print(PN_RE.ix[0:3,-1,'601318.XSHG'])  # 三者中最好的访问方式,
# 0:3表示第一至第三列，-1表示(时间)最后一天,'601318.XSHG'表示股票



# 4.处理缺失值：
print(PN.isnull().values.any())  # 检查是否有缺失值,返回Ture或者False(有缺失值返回ture)
if PN.isnull().values.any():
    PN.fillna(method='ffill',inplace=True)  # 向前填充
print(PN.isnull().values.any())



# 5.多维数据计算与合并
import talib as ta
import talib.abstract as abstract
df_ma = pd.DataFrame({name: ta.abstract.MA(value, 5) for name, value in PN.iteritems()})
# PN.iteritems()遍历三维数据每一个表格，({name:...为字典生成式
# 计算每只股票的5日均线，并且合并成DataFrame
print(df_ma.tail())

# 计算每只股票的macd, 然后合并成MultiIndex：
pn_macd = pd.Panel({name: ta.abstract.MACD(value) for name, value in PN.iteritems()})
df_macd = pn_macd.transpose(2,1,0).to_frame().head()
print(df_macd)

# 用stack()将DataFrame转换multiIndex，再将两个multiIndex合并(添加一列)：
df_macd['ma'] = df_ma.stack()
print(df_macd)


