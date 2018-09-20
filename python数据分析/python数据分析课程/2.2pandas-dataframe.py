# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 21:35:50 2018

@author: 肖
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame 

'''5.DataFrame:DataFrame是一个【表格型】的数据结构，可以看做是【由Series组成的字典】（共用同一个索引）。
DataFrame由按一定顺序排列的多列数据组成。设计初衷是将Series的使用场景从一维拓展到多维。DataFrame既有行索引，也有列索引。'''

# 5.1.1 DataFrame的创建---通过字典创建
'''最常用的方法是传递一个字典来创建,DataFrame以字典的键作为每一【列】的名称，以字典的值（一个数组）作为每一列
此外，DataFrame会自动加上每一行的索引（和Series一样）。同Series一样，若传入的列与字典的键不匹配，则相应的值为NaN。'''

dic = {'name':['张三','石六','Sara'],'age':[22,33,18],'sex':['male','female','male']}  # 定义一个字典
df = DataFrame(dic)  # index会自动补齐
df = DataFrame(dic,columns=['name','age','sex','salary'])  # 设置列名

# 案例2：DataFrame的创建---通过字典创建
df1 = DataFrame(data = {"Python":[99,101,120,78],
                        "数学":[120,136,141,149],
                        "汉语":[123,114,130,117]}, index = list("abcd"))

df1 = DataFrame(data = {"Python":[99,101,120,78],
                        "数学":[120,136,141,149],
                       "汉语":[123,114,130,117]}, index = list("abcd"), columns = ["Python","数学","汉语","英语"])

df1.values   # 输出值
df1.columns  # 输出列索引
df1.index    # 输出行索引
df1.shape    # 输出Dataframe形状

# 5.1.2 DataFrame的创建---通过numpy.ndarray创建
data = np.random.randint(0,150,(5,4))  # 创建一个numpy.ndarray
df2 = DataFrame(data = np.random.randint(0,150,(5,4)),
                columns=['语文','数学','Python','物理'],
               index = list('ABCDE'))  # 可以换成['A','B','C','D','E']
df2.shape  # 输出df2的形状
df2.values # 输出df2的值


'''5.2 DataFrame的索引'''
# 5.2.1通过属性的方式:
df2.Python  # Python是列的一个名称，如果出现行名称例如df2.A则直接报错

# 5.2.2通过类似字典的方式:
df2['Python']  # pandas.core.series.Series格式
df2[['Python']]  # 加上一个中括号，返回的结果会产生稍微的区别；pandas.core.frame.DataFrame格式
df2[['数学','Python']]  # 如果要查询多列需要再加上一个中括号，这个与前面的series是一样的

# 5.2.3对行进行索引  .loc[]和.iloc[]
'''使用.loc[]加index来进行行索引
   使用.iloc[]加整数来进行行索引'''

df2.loc['A','Python']  # 返回具体的值！(A行python列)
df2['Python']['A']  # #先找列，就可以直接使用[]
df2.loc['A']['Python']  # 返回具体的值！(A行python列)
df2.loc['A'].loc['Python']  # 返回具体的值，与上面完全一样
df2.loc['A']  # 返回一行的数据，得到一个series
df2.loc[['A','B']]  # 同时查找两行，需要再添加一个中括号，得到一个DataFrame
df2.loc['A':'C']  # 查找第A行至第C行，得到一个DataFrame

df2.iloc[0]  # 检索第一行
df2.iloc[3]  # 检索第四行
df2.iloc[0,1] # 检索第一行，第二列 ！
df2.iloc[[0,1]] # 同时检索两行，同样是再加上一个中括号
df2.iloc[0:3] # 检索第1，2，3行
df2.iloc[0:3:2] # 检索第1，3行，步长为2



'''6.DataFrame的运算 同Series一样：在运算中自动对齐不同索引的数据如果索引不对应，则补NaN'''
# 6.1 DataFrame之间的运算
df1 = DataFrame({'Python':[119,120,110],'数学':[130,118,112],'英语':[90,137,99]},
                index=['张三','王五','李四'])
df2 = DataFrame(data = np.random.randint(0,150,size = (4,4)),
                index = ['张三','王五','李四','Michael'],
               columns = ['Python','数学','物理','英语'])

df1 + 10  # dataframe的每一个元素会加10!
df1 + df2  # dataframe相加会自动对准index和column，两者相同则自动相加，对不上的则自动补齐NAN! 
df1.add(df2,fill_value=0)  # 使用这种方法可以使索引对不上者不会直接返回NAN，而是自动填上0使之进行计算
df2.add(df1,fill_value=0)  # 结果与上面完全一样

'''下面是Python 操作符与pandas操作函数的对应表：
+           add()
-           sub(), subtract()
*           mul(), multiply()
/           truediv(), div(), divide()
//          floordiv()
%           mod()
**          pow()
求平均值     mean()     
求方差       std()'''

# 6.2 DataFrame与series的运算!!!---好好理解这里是怎么回事,有一定难度。
df1 = DataFrame({'Python':[120,120,110],'数学':[150,118,112],'英语':[100,137,99]},
                index=['张三','王五','李四'])
s = df1['Python']  # type(s)；得到series
df1.add(s,axis=0)    # axis=0列的方向进行加和的操作，没有fill_value=0，同样会涉及到索引对应相加的问题
df1.add(s,axis=1)    # axis=1行的方向进行加和的操作，但是由于index对不上，结果全为空
s = df1.iloc[0]     # 得到第一行
df1.add(s,axis=0) 
df1.add(s,axis=1)   # axis=1行的方向进行加和的操作，同样会涉及到索引对应相加的问题
df1.add(s,axis='index')    # 效果同axis=0
df1.add(s,axis='columns')  # 效果同axis=1

'''
axis=0列的方向进行加和的操作,行滚动,将数字排成一行一行组成的一列
axis=1行的方向进行加和的操作,列滚动,将数字排成一列一列组成的一行
总结：DataFrameh和Series在进行运算的时候，严格的注意 axis（轴）
Series是一维的数据   DataFrame是二维的数据'''



'''7.DataFrame处理丢失数据
有两种丢失数据：None; np.nan(NaN)；none不可以计算，np.nan是可以进行计算的
但是在pandas中None与nan均视作nan参与到计算中'''
import numpy as np
import pandas as pd
from pandas import Series,DataFrame



# 7.1 含None的计算；None是Python自带的，其类型为python object。因此，None不能参与到任何计算中。
n1 = np.array([1,2,None])
n2 = np.array([1,2,np.nan])
n3 = np.array([1,2,3])
n1.sum()    # none不可以计算
n2.sum()   # np.nan可以计算,其结果均为nan
n3.sum()

df = DataFrame({'Python':[88,104,113],'数学':[118,132,119]},
               columns = ['Python','数学','英语'])
df['英语'] = [80,None,90]
df.mean(axis = 0)   # 行滚动，求列平均
df.mean(axis = 1)   # 列滚动，求行平均

# 7.2 pandas中计算None和nan均视作为nan参与到计算中，也即视作0
df = DataFrame([10,20,57,None,np.nan], index = list("abcde"), columns = ["Python"])
df.sum()

# 7.3 测试程序运行时间
import time
t0 = time.clock()
np.arange(1e5,dtype=int).sum()
t1 = time.clock()
print("Total running time: %s s" % (str(t1 - t0)))

'''7.4 判断函数-isnull()和notnull()'''
df = DataFrame({'Python':[88,104,113],'数学':[118,132,119]},
               columns = ['Python','数学','英语'])
df.loc[2,'英语'] = 127

# 7.4.1判断元素是否为空
pd.isnull(df)   # 如果为空则返回true,与下面的结果是完全相同的
df.isnull()     # 如果为空则返回true
df.notnull()    # 如果为空则返回false，不为空返回true
~df.notnull()   # 如果为空则返回true
df.isnull().all(axis = 0)  # 判断整列,全为空则返回true
df.isnull().all(axis = 1)  # 判断整行,全为空则返回true
df.isnull().any(axis = 0)  # 判断整列,有一个空则返回True
df.isnull().any(axis = 1)  # 判断整行,有一个空则返回True
df.loc[df.isnull().any(axis = 1)]  # 过滤

# 7.4.2判断整列/整行是否为空，全部不为空则返回true，有空则返回false
df.notnull().all(axis = 0)   # 判断整列
df.notnull().all(axis = 1)   # 判断整行
df[df.notnull().all(axis = 1)]  # 将不为空的全部输出

# 7.4.3判断整列/整行是否为空，只要有一个不为空就返回true，全部为空则返回false
df.notnull().any(axis = 0)   # 判断整列
df.notnull().any(axis = 1)   # 判断整行
df[df.notnull().any(axis = 1)]

# 7.4.4 df.dropna判断dataframe是否全部为true，并直接返回判断结果
df.dropna(axis = 0)  # 直接返回不为空的行，默认值是how = 'any'
df.dropna(axis = 1)  # 直接返回不为空的列，默认值是how = 'any'

df.dropna(how = 'all')  # how = 'all'是判断方式，表示只要有一个元素为true则直接返回该行；都为空才不用输出
df.dropna(axis = 0,how = 'all')
df.dropna(axis = 1,how = 'all')

# 7.4.5 元素的具体值判断方法
cond = (df >= 10)
cond = (df >= 10).any(axis = 0)
cond = (df >= 10).all(axis = 0)
cond = (df >= 10).any(axis = 1)
cond = (df >= 10).all(axis = 1)

'''7.5 缺失值的处理 df.fillna(method='ffill'/'bfill')'''
df.fillna(50)  # 为空的元素全部填充为50

# method='ffill'是向前填充
df.fillna(method='ffill',axis=0)  # 列填充
df.fillna(method='ffill',axis=1)  # 行填充

# method='bfill'是向后填充
df.fillna(method='bfill',axis=0)  # 列填充
df.fillna(method='bfill',axis=1)  # 行填充

# 7.4.6 判断元素ture 和 false ，与上面不同的是不是判断为空的方法（np.nan）,不过思考过程是完全类似的
df.any()
df.any(axis=1)



'''8.创建多层索引(多个索引)'''
import pandas as pd
import numpy as np
from pandas import Series,DataFrame

# 8.1 Series的多层索引创建
s = Series(data = np.random.randint(0,150,size = 6),
           index =[['张三','张三','李四','李四','Michael','Michael'],['期中','期末','期中','期末','期中','期末']])

# 8.2 dataframe的多层行索引创建--隐式构造：给DataFrame构造函数的index参数传递两个或更多的数组
df = DataFrame(data = np.random.randint(0,150,size = (6,3)),
               columns = ['Python','Java','PHP'],
              index = [['张三','张三','李四','李四','Michael','Michael'],['期中','期末','期中','期末','期中','期末']])
# ['张三','张三','李四','李四','Michael','Michael']是第一层的索引
# ['期中','期末','期中','期末','期中','期末']是第二层的索引

# 8.3 dataframe的多层行索引创建--显示构造
# --pd.MultiIndex.from_arrays--使用数组构建
df2 = DataFrame(data = np.random.randint(0,150,size = (6,4)),
               columns = ["Spring",'Summer','Autumn','Winter'],
               index = pd.MultiIndex.from_arrays([['张三','张三','李四','李四','Michael','Michael'],['期中','期末','期中','期末','期中','期末']]))

# --pd.MultiIndex.from_tuples--使用tuple（元组）构建,这个方法比上个方法好些，推荐使用！
df3 = DataFrame(data = np.random.randint(0,150,size = (6,4)),
               columns = ["Spring",'Summer','Autumn','Winter'],
               index = pd.MultiIndex.from_tuples([('张三','期中'),('张三','期末'),('李四','期中'),('李四','期末'),('Sara','期中'),('Sara','期末')]))

# --pd.MultiIndex.from_product--使用product构建--最简单，强烈推荐使用！！！  
# 这个方法好在写出了不同的层级，[['张三','Sara','Lisa'],['middle','end'],list('AB')]分别代表3个层级
df4 = DataFrame(data = np.random.randint(0,150,size = (12,4)),
               columns = ["Spring",'Summer','Autumn','Winter'],
               index = pd.MultiIndex.from_product([['张三','Sara','Lisa'],['middle','end'],list('AB')]))  

# 8.4 pd.MultiIndex.from_product的多层列索引--最简单，推荐使用！！！  
df5 = DataFrame(data = np.random.randint(0,150,size = (4,12)),
               columns = pd.MultiIndex.from_product([['张三','Sara','Lisa'],['middle','end'],list('AB')]),
               index = ["Spring",'Summer','Autumn','Winter'])

# 多层列索引---麻烦的写法
df6 = DataFrame(data = np.random.randint(0,150,size = (1,8)), 
               columns = [["a","a","a","a","b","b","b","b"],
                        ["期中","期中","期末","期末","期中","期中","期末","期末"],
                        ["一单元","二单元","一单元","二单元","一单元","二单元","一单元","二单元"]],
              index = ["Python"])


'''9.多层索引对象的索引与切片操作'''
# 9.1 Series的操作
s['张三']
s['张三']['期末']
s['张三','期末']  # 与上面的结果完全一样
s['期末']  # 直接查询二层索引会报错！
s.loc['期末']  # 使用loc方式则不会报错
s.loc['张三','期末']  # 使用loc方式则不会报错
# s['张三':'Michael']  # 本身是可以这样切片的，不过多层中文和英文（部分中文和部分英文不识别）索引会出现问题，这也是series的一些小bug，并不是代码的问题！
# s.loc['张三':'Michael']
s.iloc[1:3]

df3.loc['张三']
# df3.loc['期中']  # 不能直接访问二级索引
df3.loc['张三','期中']
df3.iloc[0]
df3['Spring']
df3.loc['张三']['Spring']

# 9.2 DataFrame的操作--可以直接使用列名称来进行列索引   --与之前的类似，
df3.loc['张三']
df3.loc['张三','期中']
df3.loc['李四']['Spring']
df3.iloc[0]
df3['Spring']
df3.loc['张三']['Spring']
df3['Spring']['张三','期中']
df3['Spring']['张三']

# 9.3 Dataframe多层索引切片
df7 = DataFrame(np.random.randint(0,150,size = (8,2)), 
               index = pd.MultiIndex.from_product([list("abcd"), ["期中","期末"]]), columns = ["Python","高数"])
df7.loc['a':'c']
df7['a':'c']


'''10 索引的堆（stack）！'''
'''stack()  小技巧:使用stack()的时候，level等于哪一个，哪一个就消失，出现在行里。
   unstack()'''
# 10.1 索引的堆（stack）---将列索引堆至行索引
df5.stack(level = 0)  # 第1层消失,出现在行里
df5.stack(level = 1)  # 第2层消失,出现在行里
df5.stack(level = 2)  # 第3层消失,出现在行里

# 10.2 反堆---就是第n层消失，出现在列里
df4.unstack(level=0)  # 第1层消失,出现在列里
df5.stack(level = 1)  # 第2层消失,出现在列里
df5.stack(level = 2)  # 第3层消失,出现在列里

# 10.3 unstack---将行索引堆至列索引
df7.unstack()  # 将行索引堆至列索引
df7.unstack(level = 0)  # 将行索引堆至列索引
df7.unstack(level = 1)  # 将行索引堆至列索引
df7.unstack(level= -1)  # 将行索引堆至列索引

# 聚合操作
'''小技巧:和unstack()相反，聚合的时候，axis等于哪一个，哪一个就会进行计算'''
df3.mean()
df3.mean(axis = 0)  # 列平均
df3.mean(axis = 1)  # 行平均
df3.mean(axis = 'index')  # 列平均
df3.mean(axis = 'columns')  # 行平均
df3.std(axis = 0)
df3.std(axis = 1)
df3.max(axis = 0)
df3.max(axis = 1)
df3 > 50  # 大于50的元素返回true，反之返回false
df3.where(df3 > 50)  # 不满足的数据返回nan
df3.where(df3 > 50, other = 0)  # 不满足的数据填充为0

