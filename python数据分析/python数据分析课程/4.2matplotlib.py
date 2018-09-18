# -*- coding: utf-8 -*-
"""
@author: 肖
"""

'''matplotlib.pyplot下的常用方法:
text()       mpl.axes.Axes.text()            在Axes对象的任意位置添加文字
xlabel()     mpl.axes.Axes.set_xlabel()      为X轴添加标签
ylabel()     mpl.axes.Axes.set_ylabel()      为Y轴添加标签
xticks                                       为x轴设置刻度
title()      mpl.axes.Axes.set_title()       为Axes对象添加标题
legend()     mpl.axes.Axes.legend()          为Axes对象添加图例
savefig                                      保存图片
axis         plt.axis([0,10,0,20])           设置x轴,y轴取值范围
figtext()    mpl.figure.Figure.text()        在Figure对象的任意位置添加文字
suptitle()   mpl.figure.Figure.suptitle()    为Figure对象添加中心化的标题
annnotate()  mpl.axes.Axes.annotate()        为Axes对象添加注释（箭头可选）'''

'''1.11 plot的风格和样式'''
# 1.11.1---颜色---color
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,100)
plt.plot(x,x,c='black')

x = np.linspace(0,10,100)
plt.plot(x,x,color='#007700')  # 用16进制来表示颜色,color='#eeffgg',几乎所有的颜色都可以调整,每两个数字分别表示三基色红绿蓝

x = np.linspace(0,10,100)
plt.plot(x,x,color=(0.1,0.2,0.7))  # 每个元组归一化到[0,1]的RGB元组,分别表示三基色红绿蓝

# 1.11.2---透明度---alpha参数 
x = np.linspace(0,10,100)
plt.plot(x,x,color = (0.1,0.2,0.7),alpha=0.6)  # alpha越小越淡,取值在[0,1]之间

# 1.11.3---背景色---facecolor
axes = plt.subplot(facecolor='green')  # 将背景色设置为绿色
axes.plot(x,x,'r')  # 在axes中进行绘制

# 1.11.4---线型/线条风格-ls/linestyle(两者均可)---线宽-linewidth
x = np.linspace(0,10,100)
plt.plot(x,np.sin(x),ls='-',linewidth=2)  # ls='-'表示实线,linewidth表示线条的宽度
plt.plot(x,np.sin(x),ls='-.',linewidth=3)  # ls='-.'表示点划线
plt.plot(x,np.sin(x),ls='steps',linewidth=4)  # ls='steps'表示画阶梯线

'''线条风格:
'-'     实线
':'     虚线
'--'    破折线
'steps' 阶梯线
'-.'    点划线  '''

# 1.11.5 不同宽度的破折线---dashes
x = np.linspace(0,10,100)
plt.plot(x,np.sin(x),dashes=[10,10,3,4,6,2])  # dashes列表中参数表示：第一条线长度,间隔；第二条线长度,间隔；第三条线长度,间隔

# 1.11.6 点型---marker与markersize参数
x = np.linspace(0,10,100)  # dashes列表中参数表示：第一条线长度,间隔；第二条线长度,间隔；第三条线长度,间隔
plt.plot(x,np.sin(x),dashes=[10,10,3,4,6,2],marker='3',markersize=5)  # marker='3':一角朝左的三脚架,markersize=5是三脚架大小

x = np.linspace(0,10,20)
plt.plot(x,np.sin(x),dashes=[10,10,3,4,6,2],marker='p',markersize=8)  # marker='p':五边形

x = np.linspace(0,10,20)
plt.plot(x,np.sin(x),dashes = [10,10,3,4,6,2],marker='.',markersize=10) # marker='.':点

x = np.linspace(0,10,20)
plt.plot(x,np.sin(x),dashes=[10,10,3,4,6,2],marker='_',markersize=30)  # marker='_'表示水平线

x = np.linspace(0,10,20)
plt.plot(x,np.sin(x),dashes=[10,10,3,4,6,2],marker='^',markersize=10)  # marker='^':一角朝上的三角形

'''marker参数:
'1':一角朝下的三脚架   '2':一角朝上的三脚架   '3':一角朝左的三脚架   '4':一角朝右的三脚架  
's':正方形  'p':五边形  'h':六边形1   'H':六边形2   '8':八边形
'.':点     'x':X      '*':星号      '+':加号      ',':像素,就是一个像素点（小点）
'o':圆圈   'D':菱形    'd':小菱形     '':'None',   ' ':None    '_':水平线  '|':水平线  
'v':一角朝下的三角形   '<':一角朝左的三角形   '^':一角朝上的三角形   '>':一角朝右的三角形
'''

# 1.11.7 多参数连用---颜色、点型、线型连起来使用(其实非常简单)
x = np.linspace(0,10,100)
plt.plot(x,np.sin(x),'r--<')  # 红色,--表示线形:破折线,<表示点型:一角朝左的三角形

# 1.11.8 更多点和线的设置
x = np.arange(0,10,1)
plt.plot(x,x,'r-.o',markersize = 10,markeredgecolor = 'green',markerfacecolor = 'purple',markeredgewidth = 5)

'''更多点和线的设置:
color或c:线的颜色     linestyle或ls:线型      linewidth或lw:线宽
marker:点型     markeredgecolor:点边缘的颜色    markeredgewidth:点边缘的宽度
markerfacecolor:点内部的颜色      markersize:点的大小'''

'''1.12 在一条语句中为多个曲线进行设置'''
# 1.12.1 多个曲线同一设置
x = np.arange(0,10,1)
plt.plot(x,x,x,2*x,color='r')
plt.plot(x,x,x,2*x,color='r',linestyle=':')

# 1.12.2 多个曲线不同设置
x = np.arange(0,10,1)
plt.plot(x,x,'g',x,2*x,'r')  # 如果此时写上color=’g‘就会报错
plt.plot(x,x,'g--o',x,2*x,'r:v')

# 1.12.3 分别设置方法---set方法---过于复杂麻烦
x = np.arange(0,10,1)
plt.plot(x,x,x,2*x)  # 结果会产生两行:[<matplotlib.lines.Line2D at 0x215655b60f0>,<matplotlib.lines.Line2D at 0x215655b62b0>]
line1,line2, = plt.plot(x,x,x,2*x)  # 我们接受这个结果,此时这两行就不会报出
line1.set_alpha(0.3)
line1.set_ls('--')
line2.set_marker('*')
line2.set_markersize(10)

# 1.12.4 使用setp()方法---这个方法非常巧妙！---推荐
x = np.arange(0,10,1)
line1,line2, = plt.plot(x,x,x,2*x)
plt.setp([line1,line2],linestyle = '--',color = 'r')

'''1.13 正弦余弦---'$\'是拉丁文的写法'''
x = np.linspace(0,2*np.pi,1000)
plt.plot(x,np.sin(x),x,np.cos(x))
plt.yticks(np.arange(-1,2),['min',0,'max'])
plt.xticks(np.linspace(0,2*np.pi,5),[0,'$\sigma/2$','$\delta$','$3\pi/2$','$2\pi$'],size = 20)   # size是设置字体的大小



'''2.2D图形'''
# 2.1 直方图---hist
'''hist参数说明：
bins:可以是一个bin数量的整数值，也可以是表示bin的一个序列。默认值为10
normed:如果值为True，直方图的值将进行归一化处理，形成概率密度，默认值为False
color:指定直方图的颜色。可以是单一颜色值或颜色的序列。如果指定了多个数据集合，颜色序列将会设置为相同的顺序。如果未指定，将会使用一个默认的线条颜色
orientation:通过设置orientation为horizontal创建水平直方图。默认值为vertical'''

x = np.random.randint(0,10,10)
plt.hist(x,bins=100)
plt.hist(x,bins=100,normed=True)
plt.hist(x,bins=100,normed=True,color='#3300ff')

# 2.2 条形图---bar()   水平条形图---barh()
x = [1,2,3,5,6]
y = [4,7,9,2,10]
plt.bar(x,y,width=0.2)

x = [1,2,3,5,6]
y = [4,7,9,2,10]
plt.barh(x,y)

# 2.3 饼图---pie()
# 2.3.1 普通饼图
x = [0.3,0.4,0.3]
plt.pie(x,labels=['A','B','C'])
plt.axis('equal')  # 这样才能画出绝对圆的圆
plt.show()

# 2.3.2 普通未占满饼图
x = [0.3,0.4,0.25]
plt.pie(x,labels = ['A','B','C'])
plt.axis('equal')
plt.show()

# 2.3.3 饼图属性设置---labeldistance,autopc,pctdistance
x = [0.2,0.15,0.15,0.1,0.1,0.2,0.1]
labels = ['USA','China','Europe','Japan','Russia','UK','India']
plt.pie(x,labels = labels,labeldistance=1.25,autopct='%1.3f%%',pctdistance=0.8)  # utopct='%1.3f%%'显示百分比,这里保留2位小数
plt.axis('equal')  # labeldistance=1.25设置标签圆心距;pctdistance=0.8设置数字圆心距
plt.show()

# 2.3.4 饼图阴影、分裂等属性设置---explode,shadow,startangle
x = [0.2,0.15,0.15,0.1,0.1,0.2,0.1]
labels = ['USA','China','Europe','Japan','Russia','UK','India']
plt.pie(x,labels = labels,labeldistance=1.3,autopct='%1.2f%%',pctdistance=0.8,
        explode =[0,0,0.2,0.1,0.3,0,0],shadow=True,startangle=0)  # explode设置每个部分的分离距离,shadow=True
plt.axis('equal')
plt.show()

'''饼图属性:
labeldistance:设置标签圆心距;  pctdistance:设置数字圆心距
color:设置每一块的颜色;   explode设置每个部分的分离距离
shadow:设置图片阴影;   utopct:设置显示百分比;  startangle:饼图旋转角度
'''

# 2.4 散点图---scatter---就是点图
x = np.random.randn(1000)
y = np.random.randn(1000)
colors = np.random.rand(3000).reshape(1000,3)
s = np.random.normal(loc = 35,scale= 50,size = 1000)  # 以35为均值,50为方差的正态分布
np.mean(s)  # 验证均值
np.std(s)   # 验证方差

plt.scatter(x,y,s=s,color=colors,marker='d')  # marker='d'表示菱形---查看上方点型有marker的详细参数介绍,s就是图形的大小



'''3.绘制图形内的文字、注释、箭头'''
import numpy as np
import matplotlib.pyplot as plt
# 3.1 图形内添加文本---text
x = np.linspace(0,2*np.pi,1000)
plt.plot(x,np.sin(x))
plt.text(s='sin(0)=0',x=0.2,y=0)  # 调用方法text，s是文本；x=0.2,y=0是坐标轴
plt.text(s="sin($\pi$)=0",x=3.14,y=0)
plt.figtext(s="sin($3\pi/2$)=-1",x=0.65,y=0.2)

# 3.2 箭头---arrstyles
plt.figure(figsize=(12,9))
plt.axis([0,10,0,20]);
arrstyles = ['-', '->', '-[', '<-', '<->', 'fancy', 'simple', 'wedge']  # 箭头样式
for i,style in enumerate(arrstyles):
    plt.annotate(style,xytext=(1,2+2*i),xy=(4, 1+2*i),arrowprops=dict(arrowstyle=style))

connstyles=["arc","arc,angleA=10,armA=30,rad=30", "arc3,rad=.2", "arc3,rad=-.2", "angle", "angle3"]
for i,style in enumerate(connstyles):
    plt.annotate(style,xytext=(6, 2+2*i),xy=(8, 1+2*i),arrowprops=dict(arrowstyle='->', connectionstyle=style));
plt.show()

# 3.3 综合绘制图片实例---注释---annotate()
x1 = np.random.normal(30, 3, 100)  # 生成标准正太分布数
x2 = np.random.normal(20, 2, 100)
x3 = np.random.normal(10, 3, 100)

plt.plot(x1,label='plot')  # 如果不想在图例中显示标签，可以将标签设置为_nolegend_
plt.plot(x2,label='2nd plot')
plt.plot(x3,label='last plot')

# 绘制图例---bbox_to_anchor指定图例边界框起始位置与宽高,ncol设置列数,mode="expand"图例框会扩展至整个坐标轴区域
plt.legend(bbox_to_anchor=(0, 1, 0.8, 0.5),  # 指定图例边界框起始位置为(0, 1),并设置宽度为0.8,高度为0.5
           loc=3,    # 设置位置为lower left
           ncol=3,   # 设置列数为3,默认值为1
           mode="expand",      # mode为None或者expand,当为expand时,图例框会扩展至整个坐标轴区域
           borderaxespad=0.5)   # 指定坐标轴和图例边界之间的间距

# 绘制注解---annotate()---help(plt.annotate)可以看看
plt.annotate("Important value",   # 注解文本的内容
             xy=(55,20),          # 箭头终点所在位置
             xycoords='data',     # 指定注解和数据使用相同的坐标系
             xytext=(5, 38),      # 注解文本的起始位置,箭头由xytext指向xy坐标位置
             arrowprops=dict(arrowstyle='->'));   # arrowprops字典定义箭头属性，此处用arrowstyle定义箭头风格



'''4.绘制3D图''' 
import numpy as np
import matplotlib.pyplot as plt            
from mpl_toolkits.mplot3d.axes3d import Axes3D   # 绘制3D图需要导这个包
# 4.1 案例1:
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)   
plt.plot(phi_m,phi_p)        
X,Y = np.meshgrid(phi_p, phi_m)  # 画三维图需要执行这个方法,np.meshgrid方法执行后发现这两个一维数据组被二维了
plt.plot(X,Y) # 这个图片可以看似为一个三维图片

Z = 0.7*X+2-np.sin(Y)+2*np.cos(3-X)  # 定义的三维函数
fig = plt.figure(figsize=(16,9))            
axes1 = plt.subplot(1,2,1,projection='3d')  # projection='3d'表示绘制三维图,必须填上           
axes1.plot_surface(X,Y,Z)

# 绘制第二张3D图:
axes2 = plt.subplot(1,2,2,projection='3d')
axes2.plot_surface(X,Y,Z,cmap='rainbow')  # 如果cmap的参数写错,返回的结果也会提示这个错误,并指出哪些参数可选!

# 4.2 案例2:
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)   
plt.plot(phi_m,phi_p)        
X,Y = np.meshgrid(phi_p, phi_m)  # 画三维图需要执行这个方法,np.meshgrid方法执行后发现这两个一维数据组被二维了

Z = 0.7*X+2-np.sin(Y)+2*np.cos(3-X)  # 定义的三维函数
fig = plt.figure(figsize=(16,9))            
axes1 = plt.subplot(1,2,1,projection='3d')  # projection='3d'表示绘制三维图,必须填上           
axes1.plot_surface(X,Y,Z)

# 绘制第二张3D图:
axes2 = plt.subplot(1,2,2,projection='3d')
axes2.set_xlabel('x-x')  # 绘制x轴标签
axes2.set_ylabel('y-y')  # 绘制y轴标签
axes2.set_zlabel('z-z')  # 绘制z轴标签
axes2.set_xticks(np.linspace(0,2*np.pi,5))  # 绘制x轴刻度
axes2.set_xticks(np.linspace(0,2*np.pi,5))  # 绘制y轴刻度
p = axes2.plot_surface(X,Y,Z,cmap='rainbow')  # 如果cmap的参数写错,返回的结果也会提示这个错误,并指出哪些参数可选!
plt.colorbar(p,shrink=0.8)  # axes2如果被接收(p)就不会再画出了,通过这种方式可以画出axes这张图片
# plt.colorbar是颜色棒,shrink设置颜色棒的缩放比例



'''5.绘制玫瑰图''' 
import numpy as np
import matplotlib.pyplot as plt  
x = np.random.randint(0,10,size=10)
y = np.random.randint(10,20,size=10)
plt.bar(x,y,width=0.5)  # width设置宽度
















             
             
