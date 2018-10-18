 # -*- coding: utf-8 -*-
"""
@author: 肖
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

'''1.Matplotlib基础知识'''
'''1.0 np.arange与np.linspace'''
np.arange(0,12)
np.arange(0,12,np.pi/2)
np.linspace(0,2*np.pi,5)

'''1.1 绘制单一曲线图---plt.plot'''
# 法1：
x = np.linspace(0,100,1000)
plt.plot(x,np.sin(x))
# 法2：
x = np.linspace(0,10,100)
y = np.sin(x)
plt.plot(x,y)
# 法3：使用匿名函数进行绘制
f = lambda x : x**2
plt.plot(x,f(x))

'''1.2 绘制包含多个曲线的图---plt.plot'''
# 法1：
x = np.arange(0,10,1)
plt.plot(x,x*2)
plt.plot(x,x/2)
plt.plot(x,x**2)
# 法2：也可以这样写---写在一起
x = np.arange(0,10,1)
plt.plot(x,x*2,x,x/2,x,x**2)

'''1.3 绘制点图---plt.scatter'''
# 法1：
x = np.random.randint(0,10,size=10)
plt.scatter(x,x**2)

# 法2：使用plot方法也可绘制点图---方法：添加参数:'o'
df = DataFrame(np.random.randint(0,150,size = (10,2)),index = list('abcdefhijk'),columns = ['Python','En'])
plt.plot(df['Python'],df['En'],'o')  # 绘制点图,df['Python'],df['En']分别表示横纵坐标

df['Math'] = df['Python'].map(lambda x : 0.8*x + np.random.randint(-20,20,size = 1)[0])   # 使用map函数新增一列
plt.plot(df['Python'],df['Math'],'o')  # 绘制点图,不加'o'结果为绘制线图

'''1.4 绘制网格线---plt.grid(True)'''
# 1.4.1 绘制普通网格线
x = np.arange(-np.pi,np.pi,0.01)   # np.pi表示圆周率
plt.plot(x,np.sin(x),x,np.cos(x))
plt.grid(True)

# 1.4.2 绘制其他的网格线
x = np.arange(-np.pi,np.pi,0.01)   # np.pi表示圆周率
plt.plot(x,np.sin(x),x,np.cos(x))
plt.grid(color = 'g',linestyle = '--',linewidth = 1)  # linestyle = '--'是设置的网格线样式,linewidth = 1是设置线条的粗细

'''1.5 同时绘制多张图及上面的综合运用---plt.subplot'''
plt.figure(figsize=(12,9))  # 设置图片的大小
axes1 = plt.subplot(1,3,1)  # 1行3列的第一个视图
x1 = np.arange(-20,20,0.1)
axes1.grid(color='r',linestyle='--',linewidth=2)  # linestyle='--'是网格线样式
axes1.plot(x,np.sin(x))

axes2 = plt.subplot(1,3,2)  # 1行3列的第二个视图
x2 = np.arange(-20,20,0.1)
axes2.grid(color='purple',linestyle ='-.',linewidth=2)
axes2.plot(x2,np.cos(x2))

axes3 = plt.subplot(1,3,3)  # 1行3列的第三个视图
x3 = np.linspace(0,100,1001)
axes3.grid(color='y',linestyle='--',linewidth=2,axis='y')  # axis='y'表示只在y轴绘制表格线
axes3.plot(x3,np.log(x3)) 

#///////////////////////////////////////////////////#
# subplot(221)写成这样也是可以的,表示2行2列的第一张视图
plt.figure(figsize=(12,9))
axes = plt.subplot(221)
x = np.arange(-20,20,0.1)
axes.grid(color = 'r',linestyle = '--',linewidth = 2)
axes.plot(x,np.sin(x))

axes2 = plt.subplot(222)
x2 = np.arange(-20,20,0.1)
axes2.grid(color = 'g',linestyle = '-.')
axes2.plot(x2,np.cos(x2))

axes3 = plt.subplot(223)
axes3.grid(color = 'blue',linestyle = '--',linewidth = 2)
axes3.plot(x,np.sin(x))

axes4 = plt.subplot(224)
x = np.arange(-20,20,0.1)
axes4.grid(color = 'r',linestyle = '--',linewidth = 2)
axes4.plot(x,np.sin(x))


'''1.6 调整坐标轴界限'''
# 1.6.1---通过plt.axis设置横纵坐标数值范围
x = np.random.randn(10)  # np.random.randn表示生成标准正太分布数
plt.axis([-5,15,-5,10])  # -5,15表示横坐标界限，-5,10表示纵坐标界限
plt.plot(x)

# 1.6.2---通过plt.axis设置横纵坐标属性
x = np.linspace(-np.pi,np.pi,100)
plt.plot(np.sin(x),np.cos(x))
plt.axis('off')  # 关闭横纵坐标

x = np.linspace(-np.pi,np.pi,100)
plt.plot(np.sin(x),np.cos(x))
plt.axis('equal')  # 横纵坐标相等型---这个是对画出的图形而言横纵坐标相等---而不是指坐标轴相等

x = np.linspace(-np.pi,np.pi,100)
plt.plot(np.sin(x),np.cos(x))
plt.axis('tight')  # 横纵坐标紧凑型

# 1.6.3---xlim方法和ylim方法设置横纵坐标数值范围
y = np.arange(0,10,1)
plt.plot(y)
plt.xlim(-2,12)  # x轴范围为-2到12
plt.ylim(2,10)   # y轴范围为2到10

# 1.6.4---设置坐标轴标签---plt.xlabel和plt.ylabel
x = np.arange(0,10,1)
y = x**2+5
plt.plot(x,y)
plt.xlabel('x',fontsize = 20,color = 'red')   # 设置x轴标签,fontsize = 20是该标签的字体大小
plt.ylabel('f(x) = x**2+5',rotation = 30,horizontalalignment = 'right')   # 设置y轴标签,rotation = 30表示字体旋转30度,

# 1.6.5---设置坐标轴标签刻度---xticks/yticks
# 案例1:
x = np.linspace(0,10,1000)
plt.plot(x,np.sin(x))
plt.yticks([-1,0,2])  # 调整y轴刻度
plt.xticks(np.arange(20))   # 调整x轴刻度

# 案例2:
x = np.linspace(0,10,1000)
plt.plot(x,np.sin(x))
plt.yticks([-1,0,1],['min',0,'max'])  # 数值[-1,0,1]分别表示为['min',0,'max']
plt.xticks(np.arange(10),list('abcdefhjik'))  # 数值np.arange(10)分别表示为list('abcdefhjik')

# 案例3:
x = np.linspace(0,10,1000)
y = np.sin(x)
plt.plot(x,y)
plt.xticks([0,1,2,3,4,5,6],list('abcdefg'),fontsize = 15,rotation = -60,color = 'red')
plt.yticks([-1,0,1],['min',0,'max'])   # -1 0 1 分别替换为 'min',0,'max'

# 案例4: np.pi的表示方法！--本节综合运用---$\pi$表示pi
x = np.linspace(0,10,1000)
y = np.sin(x)
np.arange(0,2*np.pi + 0.001,np.pi/2)  # 由于左闭右开原则,2*np.pi + 0.001以便取到2*np.pi这个值！
plt.plot(x,y)
plt.xticks(np.arange(0,2*np.pi + 0.00000001,np.pi/2),[0,'$\lambda$/2','$\\alpha$','3$\pi$/2','2$\pi$'])

# 1.6.5---设置坐标轴标签刻度---面向对象的方法---思想还是比较类似的
x = np.linspace(0,10,1000)
axes = plt.subplot()
axes.plot(x,np.sin(x))
# 设置对象的属性：
axes.set_yticks([-1,0,1])
axes.set_yticklabels(['min',0,'max'])

'''1.7---设置图片标题---plt.title'''
x = np.linspace(-np.pi,np.pi,100)
plt.plot(x,np.sin(x))
plt.title('f(x) = sin(x)',fontsize=20,loc='left',color='red')  # loc = 'left'是位置,还可选right,center

x = np.linspace(-np.pi,np.pi,100)
plt.plot(x,np.sin(x))
plt.title('f(x) = sin(x)',fontsize=20,loc='center',verticalalignment='bottom')

# 注意当有中文时fontsize应写在fontproperties的后面！
x = np.linspace(-np.pi,np.pi,100)
plt.plot(x,np.sin(x))
plt.title('正弦波',loc='center',fontproperties = 'KaiTi',fontsize=30)  # fontproperties设置字体

'''1.8---图例---legend---添加线条标题'''
# 1.8.1---一起添加图例
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10,1)
plt.plot(x,x,x,x*2,x,x/2)
plt.legend(['normal','fast','slow'])  # 参数传递需要中括号
plt.legend(['normal','_fast','slow']) # fast前加上'_'也可使这条线不添加图例

# 1.8.2---分别添加图例
x = np.arange(0,10,1)
plt.plot(x,x,label = 'normal')
plt.plot(x,x*2,label = 'fast')
plt.plot(x,x/2,label = 'slow')
plt.legend()  # 必须添加这行分别添加才能生效

# 1.8.3---分别添加图例---中间一条线不加图例
x = np.arange(0,10,1)
plt.plot(x,x,label = 'normal')
plt.plot(x,x*2)  # 不加label即可使这条线不添加图例
plt.plot(x,x/2,label = 'slow')
plt.legend()  # 必须添加这行分别添加才能生效

# 1.8.4---分别添加图例---中间一条线不加图例
x = np.arange(0,10,1)
plt.plot(x,x,label = 'normal')
plt.plot(x,x*2,label = '_fast')  # fast前加上'_'也可使这条线不添加图例
plt.plot(x,x/2,label = 'slow')
plt.legend()
plt.legend(mode='expand')  # mode='expand'表示展开图例

# 1.8.5---loc调整图例的位置
x = np.arange(0,10,1)
plt.plot(x,x,label='normal')
plt.plot(x,x*2,label='_fast')  # fast前加上'_'也可使这条线不添加图例
plt.plot(x,x/2,label='slow')
plt.legend(loc='upper center') 
plt.legend(loc=9)       # 给数字也是完全一样的  loc='upper center'
plt.legend(loc=(0,1))   # 如果给元组则表示相对位置

plt.plot(x, x, label='Normal')
plt.plot(x, x*2.0, label='Fast')
plt.plot(x, x/2.0, label='_Slow')
plt.legend(loc=(-0.1,0.9))  # 图例也可以超过图的界限

'''
其他loc的参数：
best         0     center left  6     upper right  1     center right 7     
upper left   2     lower center 8     lower left   3     upper center 9     
lower right  4     center      10     right        5     '''

# 1.8.6---bbox_to_anchor调整图例的位置---bbox_to_anchor---调整图例相对位置
x = np.arange(0,10,1)
plt.plot(x,x,label='normal')
plt.plot(x,x*2,label='_fast')  # fast前加上'_'也可使这条线不添加图例
plt.plot(x,x/2,label='slow')
plt.legend(bbox_to_anchor=(0,1))     # 如果给元组则表示相对位置
plt.legend(bbox_to_anchor=(0.5,1))   # 如果给元组则表示相对位置
plt.legend(bbox_to_anchor=(0.8,1))   # 如果给元组则表示相对位置
plt.legend(bbox_to_anchor=(0,1,1,0)) # 前面两个参数是图例的坐标，后面两个参数是图例的宽高

# 1.8.7---ncol参数---调整图例的列数
x = np.arange(1,5)
plt.plot(x, x*1.5, label='Normal')
plt.plot(x, x*3.0, label='Fast')
plt.plot(x, x/3.0, label='Slow')
plt.legend(loc=0,ncol=2) # ncol控制图例中有几列,ncol=2表示图例有两列
plt.show()

'''1.9---linestyle、color、marker---修改线条样式'''
# 1.9.1 绘制线条的样式---linestyle、color、marker
y1 = np.random.normal(loc=0,scale=2,size=1000)  # 生成以0为均值,1为标准差的数
y2 = np.random.normal(loc=10,scale=1.5,size = 1000)
y3 = np.random.normal(loc=-10,scale=1,size = 1000)
plt.plot(y1,linestyle='--',color = 'green',marker = '+')  # marker = '+'表示线条的标记
plt.plot(y2,linestyle='-.')  # linestyle = '-.'线条的样式
plt.plot(y3,marker='v',color = 'cyan')  # marker = 'v'表示线条的标记

# 验证下均值与方差
np.mean(y1)  # 得到均值
np.var(y1)   # 得到方差
np.std(y1)   # 得到标准差

# 1.9.2 有时可以不要前缀---是完全一样的
plt.plot(np.random.rand(1000).cumsum(),'k')   # 'k'是线条的样式
plt.plot(np.random.rand(1000).cumsum(),'k--')  
plt.plot(np.random.rand(1000).cumsum(),'k.')
plt.plot(np.random.rand(1000).cumsum(), '--',label='four')
plt.plot(np.random.rand(1000).cumsum(),'-.',label='five')
plt.plot(np.random.rand(1000).cumsum(),'--',label='six')
plt.legend(loc='best')

'''1.10---保存刚才的图片---保存图片---plt.savefig'''
y1 = np.random.normal(loc  = 0 ,scale= 1,size=100)
y2 = np.random.normal(loc = 10,scale= 1.5,size = 100)
y3 = np.random.normal(loc= - 10 ,scale= 1,size = 100)

plt.plot(y1,linestyle = '--',color = 'green',marker = '+')
plt.plot(y2,linestyle = '-.')
plt.plot(y3,marker = 'v',color = 'cyan')

plt.savefig('fig1.jpg',facecolor='red',dpi=100)  # dpi是设置图片的分辨率，默认值100
plt.savefig('fig1.png',facecolor='red',dpi=100)
plt.savefig('fig1.pdf',facecolor='red',dpi=100)
plt.savefig('fig1.ps',facecolor='red',dpi=100)
plt.savefig('fig1.eps',facecolor='red',dpi=100)  

'''  plt.savefig说明：
filename   含有文件路径的字符串或Python的文件型对象。图像格式由文件扩展名推断得出，
例如，.pdf推断出PDF，.png推断出PNG （“png”、“pdf”、“svg”、“ps”、“eps”……）
facecolor  图像的背景色，默认为“w”（白色） 
dpi        图像分辨率（每英寸点数），默认为100'''


'''1.11---综合运用'''
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,2*np.pi,20)
y = np.sin(x)
fig = plt.figure(facecolor='green')  # 保存图片里面绿色
axes = fig.add_subplot(1,1,1,facecolor = 'green')  # 添加子视图add_subplot
axes.plot(x,y,color = 'red',linestyle = '-.',label = 'sin',marker = '*',markersize = 15)
axes.plot(x,np.cos(x),color = 'purple',linestyle = '--',label = 'cos')
axes.legend(loc = 10)  # center:10 -- 参见前面 
plt.savefig('./pic3.jpg',dpi = 100,facecolor = 'green') # 保存图片









