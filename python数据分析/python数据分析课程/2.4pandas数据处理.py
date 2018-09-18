# -*- coding: utf-8 -*-
"""
@author: 肖
"""

import numpy as np
import pandas as pd
from pandas import Series,DataFrame

'''1、删除重复元素  duplicated和drop_duplicates函数'''
df = DataFrame({'color':['red','white','red','green'],'size':[10,20,10,30]})
df.duplicated()  # 使用duplicated()函数检测重复的行，返回元素为布尔类型的Series对象，每个元素对应一行，如果该行不是第一次出现，则元素为True
df.drop_duplicates()  # 使用drop_duplicates()函数删除重复的行

df2 = pd.concat((df,df),axis = 1)  # 合并
'''df2.duplicated()     # 会报错，因为存在有相同的列名
df2.drop_duplicates()   # 会报错，因为存在有相同的列名'''
 


'''2. 映射   包含三种操作：
1.replace()函数：替换元素; 2.最重要：map()函数：新建一列; 3.rename()函数：替换索引'''

# 2.1 replace()函数
color = {'red':10,'green':20}   # 设置键值对用于替换
df.replace(color,inplace=True)  # replace可以传入一个字典，通过键值对进行替换
df.replace(10,15,inplace=True)  # 将10换成15
df.loc[1] = np.nan  # 添加空值
v = {np.nan:0.1}   # 键值对，空值对应0.1
df.replace(v)  # 进行替换

# 2.2 map()函数  使用map()函数，由已有的列生成一个新列适合处理某一单独的列。
df = DataFrame({'Python':[75,80,146,35],'Java':[100,136,106,126],'PHP':[52,53,51,2],
                'HTML':[3,132,72,0]},index=['张三','旭日','阳刚','木兰'])
print(df)
v = {75:90,80:100,146:166,35:55}   # 设置键值对用于替换
df['Go'] = df['Python'].map(v)  # dataframe生成新列‘Go’,后面是隐射条件---字典
print(df)
df['C'] = df['Go'].map(lambda x : x -40)  # dataframe生成新列‘C’,后面是隐射条件---匿名函数
print(df)

def mp(x):  # 定义函数用于隐射
    if x<40:
        return('不及格')
    elif x<60:
        return('良好')
    else:
        return('优秀')
        
df['score'] = df['C'].map(mp)  # dataframe生成新列‘score’,后面是隐射条件---普通函数
print(df)

# max([1,2,10])
# df['score2'] = df['C'].map(max)  # 运行这个会报错！

# transform()和map()使用方法是一样的---了解即可上面是关键
df['score2'] = df['C'].transform(mp)
print(df)

df['C'] = df['C'].map(lambda x : x*2)  # dataframe更新列‘C’,后面是隐射条件---匿名函数
print(df)

# 2.3  rename()函数：替换索引
# 2.3.1 第一次改动
inds = {'张三':'Zhang Sir','木兰':'MissLan'}   # 设置键值对用于替换索引 
df.rename(index = inds)

# 2.3.2 第二次改动并保存
def cols(x):
    if x == 'PHP':
        return 'php'
    if x == 'Python':
        return '大蟒蛇'
    else:
        return x
    
inds = {'张三':'Zhang Sir','木兰':'MissLan'}
df.rename(index = inds,columns = cols,inplace=True)  # index = inds更改索引名称，columns = cols更改列名称
  


'''3. dataframe常用计算！！'''
# 3.0 dataframe的删除
df.drop(df.index[[1,2]])
df.drop(df.columns[3],axis=1)
df.drop(df.columns[[3,4]],axis=1)
df.drop('Zhang Sir')
df.drop(['Zhang Sir','MissLan'])
df.drop('php',axis=1)

# 3.1 dataframe常用的计算！
print(df)
df.describe()  # 描述性统计量
df.std()  # 求方差
df.std(axis=1)  # 求方差
df.std(axis=1).mean()  # 求方差的平均值
# df.abs()  # 求绝对值
np.abs([-1,-2,3])  # 同样求绝对值
df.mean()   # 求平均值
df.mean(axis=1)  # 求平均值
df.drop(['score','score2'],axis = 1,inplace=True)  # 删除两列
cond=np.abs(df)>df.std()*5
df[cond].dropna(axis=1)  # 删除含有nan的列

# 3.2 dataframe的且运算和或运算（dataframe中的元素全部为布尔值）
cond=np.abs(df)>df.std()*5
cond2 = np.abs(df) > df.mean()*1.2
print(cond,cond2)
cond&cond2  # dataframe且运算
cond|cond2  # dataframe或运算

# 3.3 dataframe一维化与互换
df2 = df.stack()  # dataframe一维化
df2 = df.stack().unstack(level = 0)  # dataframe行列互换，原来4*6的矩阵换成6*4的矩阵，当然索引和列名也会随之改变

cond.any(axis = 1)  # 按行检索，如果存在一个ture就返回true
cond.all(axis = 1)  # 按行检索，只有一行全为true才能返回true
drop_index = df[cond.any(axis = 1)].index

'''3.4 例题：去掉满足以下条件的行，其中任意一个元素的绝对值大于3倍标准差!!!'''
n = np.random.randn(10000,3)  # 生成标准正态分布
df = DataFrame(n)  # 转换为dataframe格式
cond = np.abs(df)>df.std()*3
# cond.isnull()
# cond[0][0] = np.nan
# cond.isnull()  # 为空则返回true，反之则返回false
cond.isnull().any()   # 有一个为空则返回ture
cond.isnull().any(axis=1)   # 有一个为空则返回ture（对行而言）
cond.notnull().any()   # 有一个不为空则返回ture
cond.sum()  # 通过相加可以看出有几个不为true，第一、二、三列分别有几个不为false
cond.sum(axis=1)

df[cond.notnull().any(axis=1)]   # notnull是判断为不为空的，false当然是不为空，只有np.nan是空值
df[cond.any(axis=1)]    # 通过这种方式才能判断true和or，这也是需要去掉的行了
cond.any(axis=1).index  #  RangeIndex(start=0, stop=10000, step=1)
drop_index = df[cond.any(axis=1)].index
df.drop(drop_index)  # 9921 rows x 3 columns


# 法二---通过交集差集的计算得到
index = cond.index  # RangeIndex(start=0, stop=10000, step=1)
index_1 = df[cond.any(axis=1)].index
index_2 = index.difference(index_1)
print(df.drop(index_2))
list(index_2)
df.loc[list(index_2)]  # 得到结果

'''这本身是list的求交集，并集，差集的方式，现在也可以用于这里
index.union(index_1)  # 求并集
index.intersection(index_1)  # 求交集
index.difference(index_1)  # 求差集  前面的-后面的'''



# -*- coding: utf-8 -*-
"""
@author: 肖
"""

import numpy as np
import pandas as pd
from pandas import Series,DataFrame

'''4. 排序'''
# 4.1 take与loc/iloc的差别---take可以理解为提取
df = DataFrame(np.random.randint(0,150,size = (4,4)),
               columns=['Python','Java','PHP','HTML'],
               index = ['张三','旭日','阳刚','木兰'])
df.loc['阳刚']  # 取出第三行
df.iloc[2]  # 取出第三行
df.iloc[[2,3]]  # 取出第二、三行
df.take([2])  # 取出第三行
df.take([2,3,1])  # 取出第三行,take也可以调整输出的顺序

# 4.2 排序
indices = np.random.permutation(4)  # 自动排序
df.take(indices)  # 此时得到了重新排列的数据
# sort排序
df.sort_index(by = 'PHP',axis = 0,ascending = True)
df.sort_values(by = 'PHP',axis = 0,ascending = True)
df.sort_values(by = ['PHP','Java'],axis = 0,ascending = True)  # PHP为第一要素,Java为第二要素,升序排列
df.sort_values(by = ['张三'],axis = 1,ascending = True)  # 按行排序

# 4.3 随机抽样
df2 = DataFrame(np.random.randn(10000,3))
indices = np.random.randint(0,10000,size = 10)   
type(indices)
df2.take(indices)  # 随机抽样的结果



'''5. 数据聚合【重点】！！！'''
# 5.1 groupby分类
df = DataFrame({'color':['red','white','red','cyan','cyan','green','white','cyan'],
                'price':np.random.randint(0,8,size = 8),
                'weight':np.random.randint(50,55,size = 8)})  # 字典式创建dataframe
df.groupby(['color'])  # 按照color分类
df_sum = df.groupby(['color']).sum()  # 按照color分类然后进行计算--求和

# 如果使用一个中括号['weight']或['price']会得到一个series
df_sum_weight = df.groupby(['color'])[['weight']].sum()  # 按照color分类然后进行计算--求和--并且只输出weight列
df_price_mean = df.groupby(['color'])[['price']].mean()  # 按照color分类然后进行计算--求平均--并且只输出price列

# 5.2使用pd.concat()级联！！！
pd.concat([df,df_sum_weight],axis=1)
pd.concat([df_price_mean,df_sum_weight],axis=1)

# 5.3使用df.merge()级联---只有dataframe之间才能df.merge！！！
# 将df的'color'列与df_sum_weight的索引连接起来,suffixes=['','_sum']是给另外一组相同的列名重新命一次名：
df_sum_sum = df.merge(df_sum_weight,left_on='color',
                  right_index=True,suffixes=['','_sum'])  
# 将df的'color'列与df_sum_mean的索引连接起来,suffixes=['','_平均']是给另外一组相同的列名重新命一次名：
df_r_mean = df.merge(df_price_mean,left_on='color',
                    right_index=True,suffixes=['','_平均'])

# 5.4 sort_index的使用
df_r_mean.index
df_r_mean.take([2,3])
df_r_mean.sort_index(inplace=True)  # 重新调整df_r_mean的顺序sort_index



'''6.高级数据聚合'''
# 6.1 使用transform和apply实现相同功能
df.groupby(['color']).sum()
df.groupby('color').transform(sum)  # 结果虽然一样,但格式还是与第一种有些不同
df.groupby('color').apply(sum)  # 结果虽然一样,但格式还是与上面两种方法有些不同
df.groupby('color')[['price','weight']].apply(sum)  # 挑选['price','weight']这两列进行输出

# 6.2 map添加新列
df['columns'] = df['color'].map(lambda x : x*2)
df.drop('columns',axis=1,inplace=True)







