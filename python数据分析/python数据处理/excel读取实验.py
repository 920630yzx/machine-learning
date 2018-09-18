# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:18:47 2018

@author: 肖
"""

#excel读取
import xlrd
workbook = xlrd.open_workbook('Table.xlsx')  # 打开文件
workbook.sheet_names() # 获取所有sheet
sheet1_name= workbook.sheet_names()[0]  # 获取第一个sheet，即sheet1
sheet1 = workbook.sheet_by_name('Sheet1') #根据sheet索引或者名称获取sheetq全部内容
print(sheet1.name) #获取sheet1名称
print(sheet1.nrows)  #获取sheet1行数  sheet1.nrows也可
print(sheet1.ncols)  #获取sheet1列数  sheet1.ncols也可
rows = sheet1.row_values(3) # 获取第四行内容
cols = sheet1.col_values(4) # 获取第五列内容
sheet1.cell(1,0) # 获取第二行，第一列内容，最佳方式
sheet1.cell_value(1,0)  # 获取第二行，第一列内容，格式稍有不同
sheet1.row(1)[0]  # 获取第二行，第一列内容，格式稍有不同

#excel读取  法二
data_1=pd.read_excel('Table.xlsx',sheetname=0,index_col=0)  #获取sheet1全部内容，以第一列为索引
data_2=pd.read_excel('Table.xlsx',sheetname=0,index_col='时间')  #获取shee,以时间列为索引，上面完全相同
a=data_1.iloc[1] # 获取第二行内容
b=data_1.iloc[:,0] # 获取第一列内容
c=data_1.iloc[0,:] # 获取第一行内容
d=data_1.iloc[[1,2,3,5],[0]] # 获取第二、三、四、六行，第一列内容
e=data_1.iloc[1:5,:]    # 获取第二至五行，全部列的内容
index_1=data_1.index  #索引赋值
print(index_1)   
 

#统计描述
data_1.mean()
data_1.std()
data_1.decribe()
round(data_1.mean(),2)  #round(80.23456, 3),即保留80.23456三位小数
data_1.diff()   #求差值
data_1.diff()[1:]  #求差值，不同的是从第二个数开始
data_1.pct_change()  #求增长率
data_1.rolling(window=40).mean()  #40日移动平均线
data_1.rolling(window=40).mean()[40:]  #[40:]表示起始日