# -*- coding: utf-8 -*-
"""
@author: 肖     
"""

'''1.图片灰度处理'''
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

img = plt.imread('G:/python doc/spyder picture/little.jpg')
img.shape
plt.imshow(img)

# 法1：求第三个维度的最大值或者最小值
img2 = img.min(axis = -1)    # img.min或者img.max
plt.imshow(img2,cmap = plt.cm.gray)

# 法2：求第三个维度的平均值
img3 = img.mean(axis = 2)
plt.imshow(img3,cmap = 'gray')

# 法3：加权平均法
w = np.array([[0.299,0.587,0.114]])  # 得到1*3的数组
w = np.array([0.299,0.587,0.114])  # 红绿蓝，肉眼对颜色敏感度不同，相加等于1；得到3*1的数组，注意区别。
img5 = np.dot(img,w)
plt.imshow(img5,cmap = 'gray')



'''2.设置标题'''
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-2*np.pi,5*2*np.pi,500)
y = np.e**x

axes = plt.subplot(1,3,1)
axes.plot(x,y)
axes.set_title('sin(x--1)')

axes = plt.subplot(1,3,2)
axes.plot(x,y)
axes.set_title('sin(x--2)')

axes = plt.subplot(1,3,3)
axes.plot(x,y)
axes.set_title('sin(x--3)')

plt.suptitle('sin(x)',fontsize = 15,color = 'red')  # super 超级,设置主标题



'''3.图形内的文字'''
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,2*np.pi,100)
y = np.sin(x)
plt.plot(x,y)
plt.text(3,1,'sin(0) = 0',fontsize=15,color='red') # text方法中，x，y代表是坐标值

plt.plot(x,y)
# figtext方法中代表相对值，图片宽高一个单位
plt.figtext(0.2,0.5,'sin(0) = 0',fontsize=15,color='r')  



'''4.annotate注释'''
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,15)
y = np.random.randint(10,15,size = 15)
y[8] = 30
plt.plot(x,y)
plt.axis([-2,18,8,35])  # 分别设置x,y轴的范围
plt.annotate(s = 'this point \nmean something',xy = [8,30],xytext = [10,32],   # xy设置箭头位置,xytext设置文本位置
             arrowprops = {'width':20,'headwidth':40,'headlength':10,'shrink':1})  # arrowprops绘制箭头

''' annotate参数:
    Key          Description
    width        the width of the arrow in points,箭头宽度
    headwidth    the width of the base of the arrow head in points,箭头前部宽度
    headlength   the length of the arrow head in points,箭头长度
    shrink       fraction of total length to 'shrink' from both ends,箭头前后比例
    ?            any key to :class:`matplotlib.patches.FancyArrowPatch '''



'''5.箭头---没必要记'''
plt.figure(figsize=(12,9))
plt.axis([0, 10, 0, 20]);
arrstyles = ['-', '->', '-[', '<-', '<->', 'fancy', 'simple', 'wedge']
for i, style in enumerate(arrstyles):
    plt.annotate(style, xytext=(1, 2+2*i), xy=(4, 1+2*i), arrowprops=dict(arrowstyle=style));

connstyles=["arc", "arc,angleA=10,armA=30,rad=30","arc3,rad=.2", "arc3,rad=-.2", "angle", "angle3"]
for i, style in enumerate(connstyles):
    plt.annotate(style, xytext=(6, 2+2*i), xy=(8, 1+2*i), arrowprops=dict(arrowstyle='->', connectionstyle=style));
plt.show()







