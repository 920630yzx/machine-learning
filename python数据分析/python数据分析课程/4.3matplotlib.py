# -*- coding: utf-8 -*-
"""
@author: 肖
"""

'''4.绘制3D图''' 
import numpy as np
import matplotlib.pyplot as plt            
from mpl_toolkits.mplot3d.axes3d import Axes3D   # 绘制3D图需要导这个包

'''4.1 案例1:绘制三维散点图---三个方向的x,y,z'''
x = np.random.randint(0,50,size = 100)
y = np.random.randint(-20,30,size = 100)
z = np.random.randint(50,100,size = 100)
fig = plt.figure(figsize=(8,6))  # 先给定一个二维图片
axes3D = Axes3D(fig)  # 将二维的图片，转换成三维图形
axes3D.scatter(x,y,z,s=400,color='red')    # axes3D对象本来3D，所以可以在对象，直接进行绘制
axes3D.scatter3D(x,y,z,s=400,color='red')  # axes3D.scatter3D也是绘制三维散点图,s设置点的大小

'''4.2 案例2:绘制三维线性图'''
x = np.linspace(-20,20,200)
y = np.sin(x)
z = np.cos(x)
fig = plt.figure(figsize=(8,6))
axes3D = Axes3D(fig)
axes3D.plot(x,y,z,color='red')  # 绘制线
# 三维图形,完全和二维图形用法一致,二维图形标记xlabels,ylabels
axes3D.set_xlabel('X',color='red',fontsize=15)
axes3D.set_ylabel('Y',color='blue',fontsize=15)
axes3D.set_zlabel('Z',color='green',fontsize=15)
axes3D.set_yticks([-1,0,1])

'''4.3 案例3:绘制三维柱状图'''
l = np.arange(0,10)
fig = plt.figure(figsize=(16,12))
axes3D = Axes3D(fig)

# axes3D.bar(left, height, zs=0, zdir='z', *args, **kwargs)
for i in range(3):
    h = np.random.randint(0,30,size = 10)
    axes3D.bar(l,h,zs=i,zdir='x')   # zs设置第几张柱状图,zdir='x'表示以x轴为主
    
axes3D.set_xticks([0,1,2])
axes3D.set_xticklabels(['2016年','2017年','2018年'],fontproperties='FangSong',fontsize = 18,rotation = 30)

'''4.4 案例4:绘制三维曲面图--同时绘制两张'''
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)   
plt.plot(phi_m,phi_p)            # 画二维图  
X,Y = np.meshgrid(phi_p, phi_m)  # 画三维图需要执行这个方法,np.meshgrid方法执行后发现这两个一维数据组被二维了
plt.plot(X,Y)   # 这个图片可以看似为一个三维图片

Z = 0.7*X+2-np.sin(Y)+2*np.cos(3-X)  # 定义的三维函数
fig = plt.figure(figsize=(16,9))            
axes1 = plt.subplot(1,2,1,projection='3d')  # projection='3d'表示绘制三维图,必须填上           
axes1.plot_surface(X,Y,Z)

# 绘制第二张3D图:
axes2 = plt.subplot(1,2,2,projection='3d')
axes2.plot_surface(X,Y,Z,cmap='rainbow')  # 如果cmap的参数写错,返回的结果也会提示这个错误,并指出哪些参数可选!

'''4.5 案例5:绘制三维曲面图---上例改进'''
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
axes2.set_xticks(np.linspace(0,2*np.pi,5))    # 绘制x轴刻度
axes2.set_xticks(np.linspace(0,2*np.pi,5))    # 绘制y轴刻度
p = axes2.plot_surface(X,Y,Z,cmap='rainbow')  # 如果cmap的参数写错,返回的结果也会提示这个错误,并指出哪些参数可选!
plt.colorbar(p,shrink=0.8)  # axes2如果被接收(p)就不会再画出了,通过这种方式可以画出axes这张图片
# plt.colorbar是颜色棒,shrink设置颜色棒的缩放比例

'''4.6 案例6:绘制三维曲面图---plot_surface--简化一下方便理解:'''
x = np.linspace(-2,2,100)
y = np.linspace(-2,2,100)
X,Y = np.meshgrid(x,y)
fig = plt.figure(figsize=(8,6))
axes3D = Axes3D(fig)
# Z和，X和Y之间随便给一个函数关系：
Z = np.sqrt(X**2 + Y**2)
# 绘制三维曲面:
axes3D.plot_surface(X,Y,Z,cmap=plt.cm.PuOr)  # plot_surface绘制曲面图

'''4.7 案例7:绘制三维混合图---子图绘制--其实例4.4,4.5已有介绍:'''
from mpl_toolkits.mplot3d.axes3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as axes3d   # 这里注意下差别
# 子视图添加，通过plt.subplot()
x = np.random.randint(0,100,size = 100)
y = np.random.randint(0,100,size = 100)
z = np.random.randint(0,100,size = 100)
plt.figure(figsize=(16,12))
axes3D=plt.subplot(1,2,1,projection='3d')  # projection指明图形类型,projection='3d'表示设置成3d图
axes3D.scatter(x,y,z)                      # 画第一个子视图

axes3D=plt.subplot(1,2,2,projection='3d')  # 画第二个子视图
X,Y,Z = axes3d.get_test_data()  # 这里使用了axes3d模块下,为我们提供的数据
axes3D.plot_surface(X,Y,Z,alpha=0.6)       # alpha设置透明度  

'''4.8 案例8:绘制等高线---axes3D.contour()'''
fig = plt.figure(figsize=(8,6))
axes3D = Axes3D(fig)
X,Y,Z = axes3d.get_test_data()  # 这里使用了axes3d模块下,为我们提供的数据
axes3D.plot_surface(X,Y,Z,alpha = 0.5)  # plot_surface绘制曲面图
# contour轮廓---等高线---建议只运行下面的一条试试
axes3D.contour(X,Y,Z,zdir='x')
axes3D.contour(X,Y,Z,zdir='x',offset=-30)  # zdir='x'表示x轴方向的等高线,offset设置偏差
axes3D.contour(X,Y,Z,zdir='y',offset=30)
axes3D.contour(X,Y,Z,zdir='z',offset=-60)



'''5.绘制玫瑰图''' 
import numpy as np
import matplotlib.pyplot as plt  
x = np.random.randint(0,10,size=10)
y = np.random.randint(10,20,size=10)
plt.bar(x,y,width=0.5)  # width设置宽度












