# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 09:39:18 2018

@author: 肖
"""

'''1.nupmy图形处理技术'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame 
fish = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/fish.png')
plt.imshow(fish)
print(fish.shape)  # (243, 326, 3)

fish2 = fish[::-1]  # 对高(243)进行颠倒
fish2.shape  # (243, 326, 3)
plt.imshow(fish2)

'''如果不理解参考下面内容:
a = np.random.randint(-100,100,size = (4,5))
a[::-1]
a[:,::-1]'''

fish3 = fish[::,::-1]  # 对长(326)进行颠倒
fish3.shape  # (243, 326, 3)
plt.imshow(fish3)

fish4 = fish[::,::,::-1]  # 对颜色(3)进行颠倒
fish4.shape  # (243, 326, 3)
plt.imshow(fish4)  # 得到颜色完全反过来的图片

fish5 = fish[::5,::5]  # 对图像元素进行抽取，这样得到的结果会非常模糊
fish5.shape  # (49, 66, 3)
plt.imshow(fish5)  

fish6 = fish.copy()  # .copy是直接复制！！！
fish6.shape  # (243, 326, 3)
fish6[80:120,80:110] = np.ones((40,30,3))  # 这样处理会使图片的部分被破坏
plt.imshow(fish6)

fish7 = fish[::,::,0]  # 把第三个维度颜色干掉---进行灰度化处理
plt.imshow(fish7,cmap = "gray")

'''2.pandas---Series''' 
# Series是增强版的numpy.ndarray,多了索引和标签
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame 

# 2.1 Series的创建---通过np.array进行创建
n = np.array([0,2,4,6,8])
s = Series(n)   # 先生成ndarray，再转换成Series
s.values  # 查看Series具体的值；这个值本身就是numpy.ndarray格式的
list(s.index)  # 输出Series的索引值
type(s)  #  pandas.core.series.Series
type(s.values)  # numpy.ndarray

# 创建新的索引值
s.index = ['张三','李四','Michael','sara','lisa']
print(s)

# 检索
s['sara']  # 输出s['sara']的values
s['张三'] = 100  # 给'张三'赋值
print(s)

# 2.2 Series的创建---由字典创建，通过该方法进行创建索引值自动转换成字典的键值对；
dic = {'a':np.random.randint(0,10,size = (2,3)),
       'b':np.random.randint(0,10,size = (2,3)),
            'c':np.random.randint(0,10,size = (2,3))}

s2 = Series(dic)
print(s2) 

# 2.3 直接创建-快速创建 （方法1的变形，快速写法）
s = Series(data = np.random.randint(0,150,size = 4),index=['语文','数学','英语','Python'])

# 2.4 Series的索引和切片  
'''显式索引：使用index中的元素作为索引值;使用.loc[]（推荐）'''
print(s['Python'])
print(s[['Python']])
print(s[['Python','数学']])  # 同时检索两个需要再加上一个中括号 如果是:s['Python','数学']会报错！

# 更好的方法: .loc[]（推荐）
s.loc['Python']
s.loc[['Python']]
s.loc[['Python','数学']]  # s.loc['Python','数学']会报错！

# 看看它们的格式有什么区别:
type(s.loc['Python'])  # numpy.int32
type(s['Python'])   # numpy.int32
type(s[['Python','数学']])  # pandas.core.series.Series
type(s.loc[['Python','数学']])  # pandas.core.series.Series

# 2.5 Series的索引和切片   
'''隐式索引：使用整数作为索引值;使用.iloc[]（推荐）'''
print(s[0])
print(s.iloc[[1,2]])  # 同时检索两个需要再加上一个中括号



'''3.Series的基本概念'''
# 3.1 series的基本属性
s = Series(data = np.random.randint(0,150,size = 4),index=['语文','数学','英语','Python'])
print(s.shape)
print(s.size)
print(s.values)
print(s.index)

# 3.2可以通过head(),tail()快速查看Series对象的样式
s = Series(data = np.random.randint(0,150,size = 10))
new_index = pd.date_range('20160101',periods=len(s),freq='D')
s.index = new_index  # 给s的索引附上新的值
s.head()  # 快速查看头五个
s.tail()  # 快速查看末尾五个
s.head(3)  # 快速查看头3个
s.tail(3)  # 快速查看末尾3个

# 3.3 检测缺失数据  ！！！
s = Series(data = ['张三','Sara',None])
print(s)
print(s[2])

# 可以使用pd.isnull()，pd.notnull()，或自带isnull(),notnull()函数检测缺失数据
pd.isnull(s)  # 缺失的数据返回true，否则返回false
s.isnull()    # 同样的，缺失的数据返回true，否则返回false
pd.notnull(s)   # 缺失的数据返回false，否则返回true
s.notnull()    # 同样的，缺失的数据返回false，否则返回true
s[s.notnull()]  # 通过这样的方法，可以只打印非缺失数据

# 3.4 给series取名，起到标识的作用
s.name = '姓名'
print(s)



'''4.Series的运算'''
# 4.1常规加减乘除：
s = Series(data = np.random.randint(0,100,size = 10))
s_1 = s + 10  # series的每一个元素均加上1
print(s_1)

s2 = Series(data = np.random.randint(0,100,size = 5))
print(s2)
print(s)
print(s + s2)  # 按照每个元素对应的索引进行相加，索引对不上的结果直接为nan！

# 4.2要想保留所有的index，则需要使用.add()函数
np.full((2,5),fill_value=10)
s.add(s2,fill_value=0)  # 这样后5个数据就自动填上0了，不会返回nan！






# -*- coding: utf-8 -*-
"""
傅里叶案例
"""

import numpy as np
from numpy.fft import fft,ifft  #导入傅里叶函数,fft,ifft你操作  #真实世界,时域ndarray;规律,频域高低
from PIL import Image
cat = Image.open('G:/anaconda/Spyder 项目/其他/图片包/cat.jpg')
cat

'''导入图片--法二
import matplotlib.pyplot as plt  # 导入画图
cat_1 = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/cat.jpg')
type(cat_1)  
cat_1.shape  
plt.imshow(cat_1)  
plt.show()'''

cat.tobytes()  # 输出字节
cat_data = np.fromstring(cat.tobytes(),dtype=np.int8)  # 转换成int类型数据,int8 == 128 #cat.tobytes()字节,8位 ----->对应最大的数字：-127 - 127
print(cat_data.shape)
print(cat_data)

cat_data_fft = fft(cat_data)  # 傅里叶转换,傅里叶转换的结果包含实数和虚数；真实数据转换成了频率、频域
print(cat_data_fft)

cond = np.abs(cat_data_fft)<1e5  # 判断条件，返回true或者false
inds = np.where(cond)    # 根据条件获取索引，获取为true的索引
cat_data_fft[inds] = 0  # 修改,将低频数据，设置为0（返回true的设置为0）

cat_data_ifft = ifft(cat_data_fft)  # 使用傅里叶进行反转
cat_data_real = np.real(cat_data_ifft)  # 获取实数
print(cat_data_real)

cat_data_result = np.int8(cat_data_real)  # 去除小数部分
print(cat_data_result.shape)
print(cat_data_result)

cat_outline = Image.frombytes(mode = cat.mode,size=cat.size,data=cat_data_result)
cat_outline  # 画出轮廓



'''pandas读取与写入'''
import pandas as pd
pd.read_csv('G:/python doc/spyder doc/type_comma')  # 文件第一列自动升级成列名
pd.read_csv('G:/python doc/spyder doc/type_comma',header=None)   # 重新制定列名

pd.read_csv('G:/python doc/spyder doc/type-.txt')
pd.read_csv('G:/python doc/spyder doc/type-.txt',sep= '-')  # sep= '-' 表示以'-'符号进行分割
pd.read_csv('G:/python doc/spyder doc/type-.txt',sep= '-',header=None)

pd.read_csv('G:/python doc/spyder doc/SMSSpamCollection')    # 报错
pd.read_excel('G:/python doc/spyder doc/SMSSpamCollection')  # 报错
pd.read_table('G:/python doc/spyder doc/SMSSpamCollection',header=None)  # table以tab键作为间隔

# 读取sql：
import sqlite3 as sqlite3   
con = sqlite3.connect('G:/python doc/spyder doc/1703.sqlite')
df = pd.read_sql('select * from weather_2017',con)  # 用sql语句读取
df2 = pd.read_sql('select "Date/Time","Weather" from weather_2017',con)  # 用sql语句读取
df2.head()
df3 = pd.read_sql('select * from weather_2017',con,index_col='Date/Time')  # index_col设置索引
pd.read_sql('select * from weather_2017',con,index_col='Date/Time').shape
# 写入sql：
con = sqlite3.connect('G:/python doc/spyder doc/1703.sqlite')
pd.read_sql('select * from weather_2017',con).shape
# con.execute("DROP TABLE IF EXISTS weather_2017")
df.to_sql('weather_2017',con,if_exists='append')  # if_exists='append'表示追加
pd.read_sql('select * from weather_2017',con).shape
df4 = pd.read_sql('select * from weather_2017',con)

# 从网上读取数据---只要给出数据网址即可
url = 'https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv'
pd.read_csv(url)





