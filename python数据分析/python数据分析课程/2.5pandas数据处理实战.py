# -*- coding: utf-8 -*-
"""
实战1：美国大选案例分析
"""

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

months = {'JAN' : 1, 'FEB' : 2, 'MAR' : 3, 'APR' : 4, 'MAY' : 5, 'JUN' : 6,
          'JUL' : 7, 'AUG' : 8, 'SEP' : 9, 'OCT': 10, 'NOV': 11, 'DEC' : 12}
of_interest = ['Obama, Barack', 'Romney, Mitt', 'Santorum, Rick', 
               'Paul, Ron', 'Gingrich, Newt']
parties = {
  'Bachmann, Michelle': 'Republican',
  'Romney, Mitt': 'Republican',
  'Obama, Barack': 'Democrat',
  "Roemer, Charles E. 'Buddy' III": 'Reform',
  'Pawlenty, Timothy': 'Republican',
  'Johnson, Gary Earl': 'Libertarian',
  'Paul, Ron': 'Republican',
  'Santorum, Rick': 'Republican',
  'Cain, Herman': 'Republican',
  'Gingrich, Newt': 'Republican',
  'McCotter, Thaddeus G': 'Republican',
  'Huntsman, Jon': 'Republican',
  'Perry, Rick': 'Republican'           
 }   # 候选人及其政党

# 读取txt文件
ele = pd.read_csv('G:/python doc/spyder doc/usa_election.txt',low_memory=False)   
# 默认low_memory=True，由于文件过大，low_memory=False是尽可能的使用内存
ele.shape
ele.head()
ele.tail()  # 输出最后5行

# 写入txt/csv文件
# ele = pd.to_csv('G:/python doc/spyder doc/usa_election.txt',low_memory=False)   

'''表中主要信息：
cmte_id:参选人id
cand_id:参选人id
cand_nm:参选人名称
contbr_nm:贡献者名字   contbr_city：所在城市   contbr_st：所在州    
contbr_zip:贡献者邮编    contbr_employer:贡献者所在公司   contbr_occupation:贡献者的职业
contb_receipt_amt:贡献者捐款金额    contb_receipt_dt:贡献者捐款日期
'''

# 1.使用map隐射函数，新建一列各个候选人所在党派party！（考虑下使用merge行吗？）
ele['party'] = ele['cand_nm'].map(parties)

# 2.1 使用np.unique()函数查看party这一列中有哪些元素!
ele['party'].unique()  # unique输出不一致的元素

# 2.2 使用value_counts()函数，统计party列中各个元素出现次数!
ele['party'].value_counts()

# 3.1 使用groupby()函数，查看各个党派收到的政治献金总数contb_receipt_amt！
ele.columns  # 获取所有的列名
ele.dtypes  # 获取所有的列名的详细情况
ele.groupby(['party'])['contb_receipt_amt'].sum()  

# 3.2 使用groupby函数，查看具体每天各个党派收到的政治献金总数contb_receipt_amt！
ele.groupby(['party','contb_receipt_dt']).sum()  # 先根据'party'进行分组，再根据contb_receipt_dt进行分组
ele.groupby(['party','contb_receipt_dt'])['contb_receipt_amt'].sum()  # 只显示['contb_receipt_amt']列

# 3.3 调整日期格式  依靠map函数进行自身的隐射处理！
def time_convert(dt): 
    day,mon,year = dt.split('-')
    return '20'+year+'-'+str(months[mon])+'-'+day
ele['contb_receipt_dt'] = ele['contb_receipt_dt'].map(time_convert)
ele.tail(3)  # 查看转换是否成功，输出最后三行
# 对日期格式调整的一些说明：
months['MAR']
s='29-MAR-11'
time_convert('29-MAR-11')

# 3.4 根据时间进行排序！
ele.sort_values('contb_receipt_dt',inplace=True)  # 根据contb_receipt_dt列进行排序

# 3.5 将根据时间进行排序的结果进行输出
ele_amt = ele.groupby(['party','contb_receipt_dt'])['contb_receipt_amt'].sum()
print(ele_amt)
ele.groupby(['party','contb_receipt_dt'])['contb_receipt_amt'].sum()['Democrat']  # 只查询'Democrat'的结果
ele.groupby(['party','contb_receipt_dt'])['contb_receipt_amt'].sum()['Democrat']['2011-10-17']  # 只查询'Democrat'在'2011-10-17'这一天的结果

# 4.使用unstack()将上面所得数据ele_amt中的party从一级索引变成列索引,unstack('party')！！！
# 此前ele_amt中的一级索引是政党，二级索引是日期
ele_amt_1 = ele_amt.unstack(level=0)  # 通过unstack(level=0)将一级索引'政党',变为列索引
ele_amt_2 = ele_amt.unstack(level=1)  # 通过unstack(level=1)将二级索引’日期',变为列索引,默认为level=1所以写成ele_amt.unstack()也是可以的
ele_amt_3 = ele_amt.unstack(level=0,fill_value=0)  # fill_value=0将np.nan自动填充为0

# 5.使用累加函数cumsum()，计算每个政党的资金获取的总额---这个图画得并不好，需要修改下
ele_sum = ele_amt_3.cumsum()  # 得到一个dataframe的结果
plt.figure(figsize=(12,9))
plt.plot(ele_sum)
plt.legend(['Democrat','Libertarian','Reform','Republican'])

# 6.使用stack()函数把party变成二级行索引,注意所有的值都不能为nan，需要填充为0
ele_amt_4 = ele_amt_3.stack(level=0)  # 只有1层列索引，所以填不填加‘level’都没用意义
ele_amt_4 = ele_amt_3.stack()  # 这也时间就变为一级索引，政党就变为了二级索引

# 7.查看候选人姓名cand_nm和政治献金捐献者职业contbr_occupation以及捐献情况contb_receipt_amt。能看出各个候选人主要的支持者分布情况
ele.groupby('cand_nm')['contb_receipt_amt','contbr_occupation'].sum()  # 得到每个后续人的情况，不过由于'contbr_occupation'职业信息无法通过sum计算，会自动被过滤掉
ele.groupby('cand_nm').sum()  # 得到每个后续人的情况

# 8.1 查看老兵主要支持谁：DISABLED VETERAN    # DISABLED VETERAN是一类人--老兵
cond = ele['contbr_occupation'] == 'DISABLED VETERAN'  # 返回一个布尔类型的Series
veteran = ele[cond]  # 查看这个职业的具体情况

# 8.2 使用value_counts()函数统计一下候选人出现的次数
veteran['cand_nm'].value_counts()  # 统计候选人出现的次数---看看谁在老兵当中最受支持
veteran['party'].value_counts()
veteran['contb_receipt_amt'].value_counts()

# 9.1 找出各个候选人的捐赠者中，捐赠金额最大的人的职业以及捐献额 
ele.groupby('cand_nm')['contb_receipt_amt'].max()  # 获取各个参选人当中获得的最大的一笔捐献金额

# 9.2 通过query("查询条件来查找这个捐献人职业")！！！  ---query()查询函数
ele.query("cand_nm == 'Obama, Barack' and contb_receipt_amt == 1944042.43")
ele.query("contb_receipt_amt == 1944042.43")  # 查询的结果当然完全一样了






'''实战2：苹果公司股价分析'''
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

app = pd.read_csv('G:/python doc/spyder excel/AAPL.csv')
app.head()
app.shape
app.dtypes  # 查看app的列类型,此时Date列是object类型

# 1.to_datetime()函数将'Date'这行数据转换为时间数据类型---这样做可以方便
app.Date = pd.to_datetime(app.Date)
app.dtypes  # 此时Date列是datetime64[ns]类型
app.tail()  # 输出最后5行

# 2.用set_index函数将'Date'列设置为列索引!!!(set_index修改列索引)
app.set_index('Date',inplace=True)
app.head()
app.shape

# 3.绘制苹果公司的股票走向
plt.figure(figsize=(12,9))
figure = plt.plot(app[['Adj Close']])



'''其他：修改列名：'''
df1 = DataFrame({'day':['Fri','Stat','Sun','Thur'],'1':[1,2,0,1],'2':[16,53,39,48],'3':[1,18,15,4],'4':[1,13,18,5],'5':[0,1,3,1],'6':[0,0,1,3]},
                index=[0,1,2,5],
                columns=['day','1','2','3','4','5','6'])
# 修改列名：
df1.rename(columns={'day':'n'},inplace=True)
df1.columns = ['A','B']

# 横坐标与列索引互换：
df1.stack()  # 先一维化
df1.stack().unstack(level = 0)  # 第一级索引转化为列名称
df1.stack().unstack(level = 0).plot(kind = 'bar')













