#datetime是python中用来管理和表示时间的模块，可用的类型有：
#class datetime.date: 表示日期, 属性有: year, month 和 day。
#class datetime.time: 表示一天内的时间, 属性有: hour, minute, second, microsecond 和 tzinfo。
#class datetime.datetime: date和time合并, 属性有: year, month, day, hour, minute, second, microsecond 和 tzinfo。
#class datetime.timedelta:(date, time, datetime)一段时间
#class datetime.tzinfo: 用于表示时区的抽象类

from datetime import datetime
print(datetime(2017, 1, 2, 3, 4, 5, 6)) #输出时间为2017-01-02 03:04:05.000006
print(datetime.now())  # 返回当前时间
print(datetime.utcnow())  # 返回当前UTC时间
print(datetime.fromtimestamp(1500000000)) # 返回输入的timestamp(从1970年1月1日0时开始所经历的秒数)代表的当地时间

# 通过给定的format将date_string转换成datetime：
print(datetime.strptime("2017-01-01 01:02:03.000007", "%Y-%m-%d %H:%M:%S.%f"))
# 通过给定的format将datetime类型转换成string：
print(datetime.now().strftime(format="%Y-%m-%d %H:%M:%S.%f"))
#format的格式详见： https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior



#class datetime.timedelta([days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]])
from datetime import timedelta
delta=timedelta(days=1, seconds=-2, microseconds=-3, minutes=-4, hours=-5, weeks=6)  # -号表示时间上的减法
print(delta.days, delta.seconds, delta.microseconds) 
#delta.days经历多少天,delta.seconds经历多少秒,delta.microseconds经历多少微秒



# timedelta与datetime可以做简单的四则运算：
now = datetime.now()
td = timedelta(1)  # 定义时间间隔为一天
print (now)
print (td)
print (now + td * 2)  # 进行简单的时间计算



#时间索引的修改
import pandas as pd
stock = pd.read_excel('G:/anaconda/Spyder 项目/test/sz50.xlsx',sheetname='600036.XSHG', index_col='datetime')
print(stock.tail()) #打印末尾5行数据
stock.index = list(map(lambda x: x-timedelta(hours=15), stock.index)) # 常用的方法来修改时间索引
print(stock.tail()) #通过这样的方式就可以修改时间索引



# 测试程序运行时间
import time
import numpy as np
t0 = time.clock()
np.arange(1e5,dtype=int).sum()
t1 = time.clock()
print("Total running time: %s s" % (str(t1 - t0)))


