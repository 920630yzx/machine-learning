# -*- coding: utf-8 -*-

"""
11.pandas的拼接操作
级联：pd.concat, pd.append
合并：pd.merge, pd.join
"""
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# 11.0其他内容：
# 定义一个dataframe的生成函数
def make_dataframe(cols,inds):
    data = {c:[c+str(i)for i in inds]for c in cols}
    return DataFrame(data,index = inds)

def make_df(inds,cols):
    dic = {key:[key+str(i) for i in inds]for key in cols}   #字典的key作为列名进行展示    
    df = DataFrame(dic,index = inds)
    return df
A = make_df([1,2],list('AB'))

# 两个字典生成式说明
dic1 = {key:[1] for key in ['A','B']}
dic2 = {key:[key+str(i) for i in [1,2]]for key in ['A','B']}
df = DataFrame(dic2,index = [1,2])


# 11.1 dataframe的级联--使用pd.concat()级联
df1 = make_df([1,2],list('AB'))
df2 = make_df([3,4],list('AB'))
print(df1)
print(df2)
pd.concat([df1,df2])  # 将两个datafram联合得到一个全新的dataframe,默认在列的方向进行级联,列索引应该相同
pd.concat([df1,df2],axis = 1)  # 在行的方向进行级联,行索引应该相同
pd.concat([df1,df2],ignore_index=True)  # 在列的方向进行级联，且忽视索引，重新设置新的索引

'''numpy的级联：
np.concatenate((df1,df2),axis = 1)   注意与pandas要区分开'''

# 11.2 dataframe的级联--使用多层索引keys进行级联
pd.concat([df1,df2],keys=['x','y'])

# 11.3 不匹配级联：不匹配指的是级联的维度的索引不一致。
# 例如纵向级联时列索引不一致，横向级联时行索引不一致
df1 = make_df([1,2,3],list('AB'))
df2 = make_df([3,4],list('BCD'))
df1
df2
pd.concat([df1,df2])  # 外连接,默认模式
'''
有3种连接方式：
外连接：补NaN（默认模式）
内连接：只连接匹配的项
连接指定轴 join_axes'''
pd.concat([df1,df2],join = 'outer')  # 外连接,默认模式,与上一行完全一致
pd.concat((df1,df2),join = 'inner',axis = 0)  # 内连接：只连接匹配的项，即行或者列的索引都相同
pd.concat((df1,df2),join = 'inner',axis = 1)  # 内连接：只连接匹配的项，即行或者列的索引都相同
pd.concat([df1,df2],join_axes=[df2.columns])  # 连接指定轴 join_axes，join_axes=[df2.columns]表示指定df2的所有列

# 11.4 使用append()函数添加
pd.concat((df1,df2))
pd.concat([df1,df2])
df1.append(df2)  # 与上一行结果完全相同



'''12 dataframe的合并---使用pd.merge()合并'''
'''merge与concat的区别在于，merge需要依据某一共同的行或列来进行合并
   使用pd.merge()合并时，会自动根据两者相同column名称的那一列，作为key来进行合并'''

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
# 12.1 一对一合并
df1 = DataFrame({'name':['张三','李四','Chales'],'id':[1,2,3],'age':[22,21,25]})
df2 = DataFrame({'sex':['男','男','女'],'id':[2,3,4],'group':['sale','search','service']})
df1
df2
pd.concat([df1,df2])  # 列级联
pd.concat([df1,df2],axis=1)  # 行级联
df1.merge(df2)  # 会根据相同的列--id列进行合并，且只会返回id相同的行{id=2，3}

# 12.2 多对一合并
df1 = DataFrame({'name':['张三','李四','Chales'],'id':[1,2,2],'age':[22,21,25]})
df2 = DataFrame({'sex':['男','男','女'],'id':[2,3,4],'group':['sale','search','service']})
df1.merge(df2)  # 会根据相同的列--id列进行合并，且只会返回id相同的行{id=2}，第一个有两个id=2对应第二个有一个id=2
# 另一个例子:
df3 = DataFrame({'age':[30,22,36],"work":["tech","accounting","sell"],"sex":["男","女","女"]}, index=  list("abc"))
df4 = DataFrame({"home":["深圳","北京","上海","安徽","山东"], "work":['tech',"tech","tech","accounting","sell"],
                "weight":[60,75,80,50,40]}, index = list("abcef"))
df3.merge(df4)
df4.merge(df3)   # 差别仅仅是顺序不同，其余都相同

# 12.3 多对多合并
df1 = DataFrame({'name':['张三','李四','张三'],'salary':[10000,12000,20000],'age':[22,21,25]})
df2 = DataFrame({'sex':['男','男','女'],'name':['张三','张三','凡凡'],'group':['sale','search','service']})
df1.merge(df2)  # 第一个dataframe中有两个'张三'对应于第二个'name'中有两个'张三'; 结果会分别对应，返回2*2=4个结果
# 另一个例子:
df5 = DataFrame({"home":["深圳","北京","上海","安徽","山东"], "work":['tech',"tech","tech","accounting","sell"],
                "weight":[60,75,80,50,40]}, index = list("abcef"))
df6 = DataFrame({"age":[28,30,22,36], 
                "work":["tech","tech","accounting","sell"],
                "sex":["女","男","女","男"]}, index = list("abce"))
df5.merge(df6)  # 2对3   2*3 = 6  所以有6个tech


# 12.4 当有多个列名相同时---key的规范化: 使用on=显式指定哪一列为key
df1 = DataFrame({'name':['张三','李四','张三'],'salary':[10000,12000,20000],'age':[22,21,25]})
df2 = DataFrame({'age':[21,18,29],'name':['张三','张三','凡凡'],'group':['sale','search','service']})
df1.merge(df2)   # 结果为空，这需要查看两列
df1.merge(df2,on = 'age')  # 当有两个列的列名均相同时，可以通过指定列名进行锁定，这里指定的是'age'，结果就只会返回age相同的行，# 'age':[21]
df1.merge(df2,on = 'age',suffixes = ['_a_','_b_'])  # 通过suffixes来指名另外一组相同的两列的列名称
# 另一个例子：
df3 = DataFrame({"age":[28,30,22,36], 
                "work":["tech","tech","accounting","sell"],
                "sex":["女","男","女","男"]}, index = list("abce"))
df4 = DataFrame({"age":[30,22,37],
                "work":["tech","leader", "sell"],
                "hobby":["dog","cat","fish"]}, index = list("abc"))
df3.merge(df4)  # 会打印出两列元素均相同的那一行
df3.merge(df4, on = "work", suffixes = ["_总部","_分部"])  # 只看work列,suffixes是给另外一组相同的列命名

'''12.5 强行连接: left_on='name',right_on='名字'强行连接,左边是name列右边是名字列 ！！'''
# 12.5.1 left_on和right_on在没有共同的属性的时候,才能使用:这个在工作经常使用这个方法来进行合并数据 
df1 = DataFrame({'name':['张三','李四','张三'],'salary':[10000,12000,20000],'age':[22,21,25]})
df2 = DataFrame({'年龄':[21,18,29],'名字':['张三','张三','凡凡'],'group':['sale','search','service']})
df1.merge(df2)  #　No common columns to perform merge on，没有相同的列
df1.merge(df2,left_on='name',right_on='名字')  # left_on='name',right_on='名字',通过这种方式强行连接起来

# 12.5.2 强行连接: 左边是age列右边是right_index=True表示索引列 ！！
df1 = DataFrame({'name':['张三','李四','张三'],'salary':[10000,12000,20000],'age':[22,21,25]})
df2 = DataFrame({'年龄':[21,18,29],'名字':['张三','张三','凡凡'],'group':['sale','search','service']},index=[22,21,25])
df1.merge(df2,left_on='age',right_index=True)

# 12.5.3 强行连接：扩展
df7 = DataFrame({"age":[28,30,22,36], 
                "work":["tech","tech","accounting","sell"],
                "sex":["女","男","女","男"]}, index = list("abce"))
s = df7[["age"]]*1000
s.columns = ["slary"]
df7.merge(s, left_index = True,right_index = True)

s = df7["age"]*1000
s.columns = ["slary"]
df7.merge(s, left_index = True,right_index = True)  # 报错---因为series不能合并(merge)

'''12.6 内合并、外合并、左合并、有合并 how'''
df1 = DataFrame({'name':['张三','李四','张三'],'salary':[10000,12000,20000],'age':[22,21,25]})
df2 = DataFrame({'age':[21,18,29],'名字':['张三','张三','凡凡'],'group':['sale','search','service']})
print(df1)
print(df2)
df1.merge(df2,how='inner')  # 内合并：只连接匹配的项,默认模式
df1.merge(df2,how='outer')  # 外合并：在内合并基础上,将匹配失败的项补充为NAN
df1.merge(df2,how='left')   # 左合并，df1全部输出取匹配df2，没有返回NAN
df1.merge(df2,how='right')  # 右合并，df2全部输出取匹配df1，没有返回NAN

'''12.7 列冲突   --- 类似于10.5.4 key的规范化'''
df1 = DataFrame({'name':['张三','李四','张三'],'degree':[120,118,149],'age':[22,21,25]})
df2 = DataFrame({'degree':[99,97,129],'name':['张三','张三','凡凡'],'group':['sale','search','service']})
df1.merge(df2,on='name')  # 指定name列
df1.merge(df2,on = 'name',suffixes=['_期中','_期末'])  # 指定name列，将列名相同的另一组标记出来






# -*- coding: utf-8 -*-
"""
案例分析：美国各州人口数据分析
@author: 肖
"""

import numpy as np
import pandas as pd
from pandas import Series,DataFrame

pop = pd.read_csv('G:/anaconda/excel/state-population.csv')
areas = pd.read_csv('G:/anaconda/excel/state-areas.csv')
abb = pd.read_csv('G:/anaconda/excel/state-abbrevs.csv')

pop.shape
pop.head()
areas
areas.head()
abb
abb.head()

# 1.合并pop表和abb表；'state/region','abbreviation'这两列名称不一样但是内容一样
pop_m = pop.merge(abb,left_on='state/region',right_on='abbreviation',how = 'inner')   # 默认how = 'inner',只连接匹配的项 
pop_m = pop.merge(abb,left_on='state/region',right_on='abbreviation',how = 'outer')   # 当然这里使用how = 'left'也可以
pop_m.shape

# 2.将abbreviation列删除,inplace=True表示删除后直接保存至pop_m
pop_m.drop('abbreviation',axis = 1,inplace=True)
pop_m.head()
pop_m.tail()

# 3.查看存在缺失数据的列。
pop_m.isnull().any()   # 判断整列,只要有一个为空就返回true
pop_m.isnull().any(axis = 1)  # 判断整行,只要有一个为空就返回true
pop_m.loc[pop_m.isnull().any(axis = 1)]  # 查看为空的行

# 4.填补空缺
condition = pop_m['state'].isnull()  
pop_m['state/region'][condition].unique()  #  .unique()排重，只有2个州，对应的州名为空

condition = pop_m['state/region'] == 'PR'     # 查询PR
pop_m['state'][condition] = 'Puerto Rico'    # 给予新的名称
condition = pop_m['state/region'] == 'USA'   # 查询'USA'
pop_m['state'][condition] = 'United State'   # 给予新的名称
pop_m.isnull().any()   # 刚才的填补操作，起作用了，州名称已经全部补齐，可以与areas进行合并了

# 5.pop_m与areas进行合并
pop_areas_m = pop_m.merge(areas,how = 'outer')
pop_areas_m.shape
pop_areas_m.isnull().any()  # 发现仍然右两列缺失了

# 6.查看缺失得area(sq.mi)
cond = pop_areas_m['area (sq. mi)'].isnull()
pop_areas_m['state/region'][cond]
pop_areas_m['state/region'][cond].unique()  #  .unique()排重，只有1个州(下面没有选择填补，而是选择直接去除缺失数据)

# 7.删除含有缺失数据的行
pop_areas_m.shape
pop_areas_r = pop_areas_m.dropna()   # dropna()默认清除缺失的行
pop_areas_r.shape
pop_areas_r.isnull().any() # 可以看见已经无缺失值了
pop_areas_r.head()

'''其他删除方法：
pop_m.drop(2)
pop_m.drop([2,3])
pop_m.drop(pop_m.index[[2,3]],inplace=True)
pop_m.drop(condition,inplace=True)
df.drop(df.index[[1,2]])  # 删除第二第三行
df.drop(df.columns[3],axis=1)  # 删除第四列
df.drop(df.columns[[3,4]],axis=1)  # 删除第四第五列
df.drop('Zhang Sir')  # 删除Zhang Sir行
df.drop(['Zhang Sir','MissLan'])  # 删除Zhang Sir行和MissLan行
df.drop('php',axis=1)'''  # 删除php列

# 8.进行分析
# 8.1 用query对dataframe进行挑选：
t_2010 = pop_areas_r.query("ages == 'total'")
t_2010 = pop_areas_r.query("ages == 'total' and year == 2010")
t_2010.shape

# 8.2 用set_index调整dataframe的索引
t_2010.set_index('state',inplace=True)  # 让'state'作为新的索引

# 8.3 计算人口密度。注意是Series/Series，其结果还是一个Series。
pop_density = t_2010['population']/t_2010["area (sq. mi)"]
type(pop_density)  # pandas.core.series.Series

# 8.4 series排序，并找出人口密度最高的五个州sort_values()
pop_density.sort_index()  # 对索引进行排序
pop_density.sort_values(inplace=True) # 对值进行排序
pop_density[:5]  # 人口密度最低的五个州
pop_density.tail()  # 人口密度最高的五个州






