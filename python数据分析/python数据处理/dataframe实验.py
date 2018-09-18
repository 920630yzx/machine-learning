import pandas as pd
import numpy as np
#1.字典转DataFrames的例子：
dict_data = {
'a' : [1, 2, 3, 4, 5],
'b' : ['L', 'K', 'J', 'M', 'Z'],
'c' : np.random.normal(0, 1, 5)}
print (dict_data)
frame_data = pd.DataFrame(dict_data, index=pd.date_range('20160101',periods=5))
print (frame_data)



#2.Series组合成DataFrame的例子：
s_1 = pd.Series([2, 4, 6, 8, 10], name='APPL') # 创建Series,'APPL'为s_1名称
s_2 = pd.Series([1, 3, 5, 7, 9], name="GOOG")
numbers = pd.concat([s_1, s_2],axis=1)  # pd.concat将Series组合成DataFrame
#若axis=0会成为10*1的Series,axis=1才会生成2列的dataframe
print (numbers)
print (type(numbers))



#3.修改dataframes的列名、索引；
print (numbers.columns) #打印列名
print (numbers.values)  #打印值
numbers.columns = ['MSFT', 'YHOO'] # 暴力修改列名  
numbers.rename(columns={'MSFT':'MSFT2'}, inplace = True)  #更好的修改方法,将'MSFT'修改为'MSFT2'
print (numbers)
numbers.index = pd.date_range("20160101",periods=len(numbers)) # 修改索引  
print (numbers)
print (numbers.values)  # 读取Dataframe的值



# 4.1访问DataFrame元素
# 读取当前目录  stock1 = pd.read_excel('sz50.xlsx',sheetname='600036.XSHG', index_col='datetime')
stock1 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600036.XSHG', index_col='datetime')
stock2 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600050.XSHG', index_col='datetime')
stock3 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='601318.XSHG', index_col='datetime')
stock4 = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='601088.XSHG', index_col='datetime')
data3=stock4.close   # 读取dataframe的close列
data4=stock4['close']  # 读取dataframe的close列,两种方法都可以
data5=stock4.close.iloc[3:6]  # 读取dataframe的close列的第四行-第六行

from datetime import datetime
symbol=['600036.XSHG','600050.XSHG','601318.XSHG']
data_dict = {}#定义字典
# 4.2循环读取
for s in symbol:
    data =  pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx', sheetname=s, index_col='datetime')
    data_dict[s] = data['close']  # 输出close收盘价那一列,输出为字典,通过循环定义字典！
    
data = pd.DataFrame(data_dict)  # 将字典转换成dataframe格式
print(data.loc[datetime(2017,1,1):datetime(2017,1,10),['600036.XSHG', '601318.XSHG']])  # 选择时间、股票进行打印
print(data.iloc[0:2,1])  # 打印第一行、第二行的第一列
print(data.iloc[[1, 3, 5] + list(range(7, 20, 2)), [0, 1]].head(20))  # 选择日期进行打印,[0, 1]表示列



#布尔索引
#与Series一样,有时候我们想过滤DataFrame根据一组标准,我们通过索引DataFrame布尔值
data['600036.XSHG'].pct_change()  # 求变化率
print(data.loc[data['600036.XSHG'].pct_change() > data['601318.XSHG'].pct_change()].head())

# 5.添加、删除列,结合DataFrames/Series 
# 当你已经有一个DataFrame的数据,这很好,但同样重要的是能够增加你的数据：
new = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600519.XSHG', index_col='datetime')
data['600519.XSHG'] = new.close  # 将new.close列添加进data['600519.XSHG']中去,data'600519.XSHG'列变化
print(new.close.iloc[3:5])
print(data.head(5))

data = data.drop('600050.XSHG', axis=1) #将600050列从data中删去
print(data.head(5))

gold_stock = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600547.XSHG', index_col='datetime')
df=pd.concat([data,gold_stock['close']], axis=1)  # 合并某一列(gold_stock['close'])
print(df.head(5))

df.rename(columns={'close':'600547.XSHG'}, inplace = True)  # 修改列名称，'close':'600547.XSHG'代表旧名称'close'换为新名称'600547.XSHG'
print(df.head(5))



#6.缺失的数据处理(NAN):作业1中也有缺失数据的处理  
#dropna()表示删除为空的行; .dropna(axis=1)表示删除为空的列;同时可以通过设置inplace=True直接修改data的值，默认是是Flase
#dropna(how="all") 表示删除所有元素为空的行;.dropna(axis=1)表示删除所有元素全部为空的列
df=pd.concat([df,data4], axis=1) # 再添加一列
A=df.isnull()  # 若datagrame中有NAN则返回true,会返回一个dataframe
A2=df[df.isnull().values==True]  # A2为存在缺少数据的行,同样会返回一个dataframe
print(df[df.isnull().values==True])  # 打印有nan的行
df_na = df.fillna(method='ffill')  # 填充,改变本来的样本(对于多列也可以分别进行填充)
print(df_na.loc['2017-04-26':'2017-05-17'])
print(df_na)



#7.dataframe分析和计算
#使用安装在内部的统计方法来计算DataFrames,我们可以对多个时间序列进行计算。执行计算的代码在DataFrames与在series上几乎一模一样。
import matplotlib.pyplot as plt

onebegin=data/data.iloc[0]  # 净值标准化,以1为标准
plt.figure(figsize=(15, 7))
plt.plot(onebegin)  # 画图 onebegin.plot()也可
plt.title("Onebegin Stock Prices")
plt.ylabel("Price")
plt.xlabel("Date")
plt.show()

print('mean:','\n',data.mean(axis=0)) # axis=0求每列均值,axis=1则求每行均值,也许简单的来记就是axis=0代表往跨行（down)，而axis=1代表跨列（across)
print('std:','\n',data.std(axis=0)) # axis=0求每列方差,axis=1则求每行方差; 然而,如果我们调用df.drop((name, axis=1),我们实际上删掉了一列,而不是一行
print(onebegin.head(10))  # 输出前10日净值变化

# 7.1将回报率标准化，然后可视化。
mult_returns = data.pct_change()[1:]  # pct_change计算每日收益率,从第二日开始计算
print(mult_returns.head(5))
norm_returns = (mult_returns-mult_returns.mean(axis=0))/mult_returns.std(axis=0) # 即求每日回报率的波动情况,只是没累加
plt.figure(figsize=(15, 7))
plt.plot(norm_returns)  # 画出该波动率
plt.hlines(0, norm_returns.index[0],norm_returns.index[-1], linestyles='dashed')
plt.show() # 对于为什么没有plt.show()也能显示图形？那是因为你用的是ipython这是一个交互式界面

# 7.2将dataframe里的数据计算移动均线，最后可视化展示出来：
data['601318.XSHG'].rolling(window=5,center=False).mean() # 计算5日均线
rolling_mean = data['601318.XSHG'].rolling(window=40,center=False).mean() # 计算40日均线
plt.figure(figsize=(15, 7))
data['601318.XSHG'].plot() # 打印601318.XSHG
print(data.iloc[:,1]) # 打印第二列,效果与上面完全一致
print(data.iloc[0:2,1]) # 打印第一行、第二行;第二列
rolling_mean.plot()
plt.title("40days Rolling Mean of 601318.XSHG")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

'''其他dataframe计算的方式:
1、对某一行进行求和： Row_sum = df.iloc[i,0：].sum()
2、对某一列进行求和：column_sum = df.iloc[:,j].sum()
3、对每一列进行求和：
        for i in df.columns:
            print(df[i].sum())'''

# 8.dataframe计算： apply()自定义函数
from pandas import DataFrame
import pandas as pd
import numpy as np
#生成DataFrame数据:  np.random.randn的取数范围：正态分布的随机样本数
df = DataFrame(np.random.randn(4, 5), columns=['A', 'B', 'C', 'D', 'E'])

#计算各行数据总和并作为新列添加到末尾: df['Col_sum']是添加列的方法
df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)  # axis=1表示计算每行,是列滚动

#计算各列数据总和并作为新行添加到末尾: df.loc['Row_sum']是添加行的方法
df.loc['Row_sum'] = df.apply(lambda x: x.sum())   # 默认axis=0表示计算每列,是行滚动

'''删除行：
odata.drop([16,17]) #删除第16行,17行
odata.drop(odata.index[[16,17]],inplace=True) #删除第16行,17行
#区别在于.drop()方法如果不设置参数inplace=True,则只能在生成的新数据块中实现删除效果,
#而不能删除原有数据块的相应行,如果inplace=True则原有数据块的相应行被删除

删除列：
del odata['date'] #删除'date'列
spring = odata.pop('spring') #.pop方法可以将所选列从原数据块中弹出,原数据块不再保留该列
withoutSummer=odata.drop(['summer'],axis=1) #删除summer列,无inplace=True固odata本身不变,新的dataframe中则每月summer列
withoutWinter = odata.drop(['winter'],axis=1,inplace=True) #当inplace=True时.drop()执行内部删除,不返回任何值,原数据发生改变

col_name.insert(col_name.index('B')+1,'D') # 在B列后面插入D
col_name.insert(1,'D') #在1列后面插入D
df.rename(columns={'close':'600547.XSHG'}, inplace = True) #修改列名称
df.set_index(['c']) #将‘c’列转换为索引
df.set_index(['c','d']).reset_index() #想要把索引还原到列中可以使用reset_index()

删除空值
df.dropna()     #将所有含有nan项的行删除
df.dropna(axis = 1)  #删除列
df.dropna(axis=1,thresh=3)  #将在列的方向上三个为NaN的项删除
df.dropna(how='ALL')        #将全部项都是nan的row删除'''






