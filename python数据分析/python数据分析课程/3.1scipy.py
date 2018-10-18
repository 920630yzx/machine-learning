# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:47:48 2018
@author: 肖
"""
'''Scipy依赖于Numpy,Scipy是高端科学计算工具包
Scipy包含的功能：最优化、线性代数、积分、插值、拟合、特殊函数、快速傅里叶变换、信号处理、图像处理、常微分方程求解器等'''

import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack  # fftpack是快速傅里叶变换的一个包

'''1.scipy图片噪音过滤---傅里叶变化操作---此方法还可以过滤音频,操作原理类似'''
# 1.读取图片
moon_data = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/moonlanding.png')
print(moon_data)
print(moon_data.shape)
print(moon_data.size)
print(moon_data.ndim)   # 维度
print(moon_data.dtype)  # 元素类型
plt.figure(figsize=(12,9))
plt.imshow(moon_data)
plt.imshow(moon_data,cmap='gray')  # cmap='gray'将图片颜色变为黑白的

# 2.使用fftpack.fft2或者fftpack.fft进行傅里叶变化操作！
moon_data_fft = fftpack.fft2(moon_data)   # fftpack.fft2对二维、三维数组进行傅里叶变化；fftpack.fft是对一维数组进行傅里叶变化

# 3.进行高频率的过滤,突兀===高频,过滤高频
condition = np.abs(moon_data_fft) > 8e2  
index = np.where(condition)     # 查找满足这一条件的索引，但是这种输出格式不好，使用argwhere更好！
index = np.argwhere(condition)  # 查找满足这一条件的索引；第一列代表纵轴，第二列代表横轴。
moon_data_fft[condition] = 0    # 将满足条件项置为0，其用意是将高频,高频的过滤掉

# 4.使用fftpack.ifft2进行反傅里叶操作，变为实域（结果多含有虚数）
moon_data_ifft = fftpack.ifft2(moon_data_fft)

# 5.去除虚数保留实数---np.real去虚保实操作
result = np.real(moon_data_ifft)
result.shape
plt.figure(figsize=(12,9))
plt.imshow(result,cmap='gray')  # cmap='gray'表示以灰色进行展示

# 直接操作(法2)---np.where方法---与上述类似---不同之处仅在于第三步
moon_data = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/moonlanding.png')
plt.figure(figsize=(12,9))
plt.imshow(moon_data)
moon_data_fft = fftpack.fft2(moon_data)  # 进行傅里叶变化操作！
moon_data_fft_r = np.where(np.abs(moon_data_fft)>3e3,0,moon_data_fft)  # np.where获取满足条件的索引
# np.abs(moon_data_fft)>3e3是满足这个条件; 0,moon_data_fft表示将moon_data_fft中满足条件的更换为0  
# 可以看出比上例过滤值更高，即过滤的数更少，得到的图片理论上相较于原始图改变的也会更少
moon_data_ifft = fftpack.ifft2(moon_data_fft_r)  # 进行反傅里叶变化操作！
result = np.real(moon_data_ifft)  # np.real进行去虚保实操作
plt.figure(figsize=(12,9))  # 画图
plt.imshow(result,cmap='gray')  # cmap='gray'表示以灰色进行展示



'''2.使用scipy求积分---integrate.quad'''
# 2.1 首先画一个圆:  X^2 + Y^2 = 1
''' 分析:
x**2 + y**2 = 4
y**2 = 1 - x**2
f(x) = (1 - x**2)**0.5 '''

f = lambda x : (1 - x**2)**0.5   # 这种写法注意一下！
x = np.arange(-1,1,0.001)   # 两种方式均可
plt.figure(figsize=(4,4))
plt.plot(x,f(x))
plt.plot(x,-f(x))

x = np.linspace(-1,1,2001)  # 两种方式均可,但这种更好！
plt.figure(figsize=(4,4))
plt.plot(x,f(x))
plt.plot(x,-f(x))

x = np.linspace(-1,1,2001)  # 两种方式均可,但这种更好！
y = list(map(lambda x:(1-x**2)**0.5,x))
z = list(map(lambda x:-(1-x**2)**0.5,x))
plt.figure(figsize=(4,4))
plt.plot(x,y)  # 此时x,y均是固定的值
plt.plot(x,z)

# 2.2 使用scipy.integrate进行积分
import scipy.integrate as integrate
pi_2,err = integrate.quad(f,-1,1)  # pi是结果，err是误差，integrate.quad本身就会返回两个值
pi = 2*pi_2
print(pi)



'''3.图片处理---Scipy'''
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

'''3.1使用scipy中的io.savemat()进行图片的保存读取,文件格式是.mat,标准的二进制文件'''
import scipy.io as spio
moon_data = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/moonlanding.png')
spio.savemat('moon.mat',{'moon':moon_data})   # 'moon.mat'是另存为的文件名称；moon_data是一组需要保存的数据，必须写成字典式才行
spio.loadmat('moon.mat')   # 使用io.loadmat()读取数据，看来是个字典，需要添加['moon']进行读取
moon = spio.loadmat('moon.mat')['moon']   #  使用io.loadmat()读取数据
plt.imshow(moon,cmap = 'gray')

'''3.2 图片处理---scipy.misc进行图片操纵   
用于操纵图片,包括：图片读取、图片保存、图片缩放、图片旋转、图片过滤...'''
# 3.2.1misc.imread读取图片---区别与plt.imread！
import scipy.misc as misc   
cat_data = misc.imread('G:/anaconda/Spyder 项目/其他/图片包/cat.jpg')
plt.imshow(cat_data)  
# misc.imshow(cat_data)   # 另一种显示图片的方式---弹框式！---不过spyder好像不支持这一功能

# 3.2.2 使用misc.imrotate进行图片的旋转
cat_data2 = misc.imrotate(cat_data,angle=90)   # 旋转90度
misc.imsave('cat2.jpg',cat_data2)   # 图片保存！
plt.imshow(cat_data2)
# misc.imshow(cat_angle)  # 另一种显示图片的方式---弹框式！

# 3.2.3 使用misc.imresize对图片进行缩放
cat_data3 = misc.imresize(cat_data,0.5)  # 缩放为原来的0.5倍
plt.imshow(cat_data3)   
cat_data3 = misc.imresize(cat_data,(100,200))
plt.imshow(cat_data3)   # 缩放为指定的大小   这里表示100*200*3的jpg格式文件
misc.imsave('cat3.jpg',cat_data3)   # 图片保存！

# 3.2.4 使用misc.imfilter对图片进行过滤
cat_data4 = misc.imfilter(cat_data,ftype='edge_enhance')  # ftype='edge_enhance'填写过滤方式
plt.imshow(cat_data4)
cat_data4 = misc.imfilter(cat_data,ftype='blur')    # 另一种过滤方式
plt.imshow(cat_data4)
cat_data4 = misc.imfilter(cat_data,ftype='detail')  # 另一种过滤方式，当然还有其他多种过滤方式，自己查看帮助ctrl+i
plt.imshow(cat_data4)

'''3.3 图片处理---scipy.ndimage
用于操纵图片,包括：图片移动、图片旋转、图片缩放、图片切割、图片模糊处理...'''
import scipy.ndimage as ndimage
plt.imshow(misc.face())   # misc.face()是misc的内置图片，该图片是一只熊
face = misc.face(gray=True)  # 灰白处理
plt.imshow(face)
plt.imshow(face,cmap = 'gray')

face = misc.face(gray=False)  
plt.imshow(face)
plt.imshow(face,cmap = 'gray')  # 这里灰白处理居然无效了

face.shape   # (768, 1024, 3)
face.std()
face.dtype  # dtype('uint8'),得到face中的元素的属性,与type(face)是不一样的

# 3.3.1 shift移动坐标:
face_shift = ndimage.shift(face,(200,0,0))  # 向下移动200距离;
# 由于face是三个维度,所以需要填三个参数(200,0,0),不然报错:sequence argument must have length equal to input rank
plt.imshow(face_shift,cmap = 'gray')  # cmap='gray'表示以灰色进行展示

face_shift = ndimage.shift(face,(200,300,0))  # 向下移动200距离，向右移动300距离
plt.imshow(face_shift,cmap = 'gray')  # cmap='gray'表示以灰色进行展示

face_shift = ndimage.shift(face,(300,0,1),mode='mirror')  # 向下移动300   mode还有多种格式可以支持,自己查看帮助文档
plt.imshow(face_shift,cmap = 'gray')  # 注意最后一个元素是1(1、5、9效果相同),这是通过改变第三个维度来对颜色进行改变,cmap='gray'表示以灰色进行展示

face_shift = ndimage.shift(face,(-300,0,4),mode='nearest')  # 向上移动300   mode还有多种格式可以支持,自己查看帮助文档
plt.imshow(face_shift,cmap = 'gray')  # cmap='gray'表示以灰色进行展示,第三个维度为4刚好不会改变

# 3.3.2 rotate旋转图片
face_rotate = ndimage.rotate(face,180)   # 旋转180度
plt.imshow(face_rotate,cmap = 'gray')

face_rotate = ndimage.rotate(face,180,axes=(-2,-3))  # axes=(-2,-3)指定旋转维度,还可以改为axes=(-1,-3),axes=(-3,-2)---不好理解自己实验吧
plt.imshow(face_rotate,cmap = 'gray')  

# 3.3.3 zoom缩放图片
face_zoom = ndimage.zoom(face,zoom = 0.05)
plt.imshow(face_zoom)
plt.imshow(face_zoom,cmap = 'gray')
 
# 3.3.4 切割图片/图片切割---这并不是scipy.ndimage中的方法，常规方法切割即可
face_mini = face[200:400,200:500]  
plt.imshow(face_mini,cmap = 'gray')



'''3.3 图片处理---ndimage---模糊图片处理(与傅里叶的方法效果有相同之处)
'''
face = misc.face(gray=True)
plt.imshow(face)
plt.imshow(face,cmap = 'gray')

# 3.3.1为清晰的图片添加噪声:
face.shape
face.std()
face.dtype
noisy_face = face.copy().astype(float)  # 转换数字格式为float型
noisy_face.dtype  
noisy_face += face.std()*0.3*np.random.standard_normal(face.shape)   # 生成随机数np.random.standard_normal的这个方法可以直接输入size格式
plt.imshow(noisy_face)  # 查看图片，发现图片变得非常粗糙

# 3.3.2 高斯滤波---ndimage.gaussian_filter过滤
face_gaussian = ndimage.gaussian_filter(noisy_face,sigma=1)  # sigma=1,高斯核的标准偏差；设置的标准差越大过滤的数越多，当然这个数需要调整看看哪些数合适
plt.imshow(face_gaussian)  # 可以发现变得均变得清晰了
plt.imshow(face_gaussian,cmap='gray')

# 3.3.3 中值滤波---ndimage.median_filter
face_median = ndimage.median_filter(noisy_face,size = 5)  # 中值滤波参数size：给出在每个元素上从输入数组中取出的形状位置，定义过滤器功能的输入；
plt.imshow(face_median)
plt.imshow(face_median,cmap='gray')

# 3.3.4 维纳滤波---mysize
import scipy.signal as signal
face_wiener = signal.wiener(noisy_face,mysize=3)  # mysize必须使奇数
plt.imshow(face_wiener)
plt.imshow(face_wiener,cmap = 'gray')


# 3.3.5 处理月球图片---对比傅里叶变换的结果---2.1章末尾
moon = plt.imread('G:/python doc/spyder picture/moon.png')
plt.figure(figsize=(12,9))
plt.imshow(moon,cmap = 'gray')

moon_gaussian = ndimage.gaussian_filter(moon,sigma=2)  # 使用高斯滤波过滤
plt.figure(figsize=(12,9))
plt.imshow(moon_gaussian,cmap = 'gray')

moon_median = ndimage.median_filter(moon,size=7)  # 使用中值滤波过滤
plt.figure(figsize=(12,9))
plt.imshow(moon_median,cmap = 'gray')

moon_wiener = signal.wiener(moon,mysize=5)  # 使用维纳滤波过滤
plt.figure(figsize=(12,9))
plt.imshow(moon_wiener,cmap = 'gray')



'''4.灰度化处理'''
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np

zhima  = plt.imread('G:/python doc/spyder picture/芝麻.jpg')
plt.imshow(zhima)

zhima_noisy = zhima.copy().astype(float)  # 调整元素的格式
plt.imshow(zhima_noisy)  # 调整元素的格式，图片发生细微的变化
np.random.standard_normal(zhima_noisy.shape).min()
np.random.standard_normal(zhima_noisy.shape).max()

zhima_noisy += zhima_noisy.std()*0.3*np.random.standard_normal(zhima_noisy.shape)   # 生成随机数np.random.standard_normal的这个方法可以直接输入size格式
plt.imshow(zhima_noisy)


# 4.1图片灰度处理---平均值法
zhima.shape
zhima_mean = zhima.mean(axis=2)  # 对最后一维求其平均值（axis=-1是完全一样的），如果不给axis则会对zhima的全部数据(1986000个数据)求平均值
zhima_mean.shape
plt.imshow(zhima_mean)
plt.imshow(zhima_mean,cmap = 'gray')  # cmap = 'gray'是进行灰白处理

# 4.2图片灰度处理---最大值法
zhima_max = zhima.max(axis=2)  # 对最后一维求最大值（axis=-1是完全一样的）
zhima_max.shape
plt.imshow(zhima_max)  
plt.imshow(zhima_max,cmap = 'gray')  # cmap = 'gray'是进行灰白处理，可以看出较上面的结果更亮了（最后1维表示颜色，值越大越亮）

# 4.3图片灰度处理---加权平均法
# red=0.299，green=0.587，blue=0.114
gravity = np.array([0.299,0.114,0.587])
zhima_gravity = np.dot(zhima,gravity)  # 矩阵相乘,用最后1维进行相乘,1*3的矩阵乘以3*1的矩阵
gravity.shape  # (3,)
zhima_gravity.shape  # (662, 1000)
plt.imshow(zhima_gravity)  
plt.imshow(zhima_gravity,cmap = 'gray') 










