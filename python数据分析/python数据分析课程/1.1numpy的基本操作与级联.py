# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:02:34 2018

@author: 肖
"""
import numpy as np
np.__version__  # 查看numpy版本

'''1.读取图片及numpy属性'''
# 1.1读取图片plt.imread与展示图片plt.imshow
import matplotlib.pyplot as plt  # 导入画图
cat = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/cat.jpg')   # 读取cat这张图片（该图片在本地）
print(cat)
type(cat)  # numpy.ndarray
cat.shape  # (456, 730, 3)  这说明图片是三维的；电影呢？当然就是四维了(x,456,760,3)
plt.imshow(cat)  # 将对应路径的图片展示出来，也就是画出来
plt.show()

# 1.2ndarray的属性（牢记！）
cat.ndim   # 维度
cat.shape  # 形状（各维度的长度）
cat.size   # 总长度
cat.dtype  # 元素类型



'''2.np.array的创建：numpy默认ndarray的所有元素的类型是相同的
如果传进来的列表中包含不同的类型，则统一为同一类型，优先级：str>float>int'''
import numpy as np
# 2.1创建数组/矩阵
l = [3,1,4,5,9,6]
n = np.array(l)
n.shape  # 得到(6,)，说明是1维的
print(n,l)  # 升级成2维数组
n2 = np.array([[3,4,7,1],[3,0,1,8],[2,4,6,8]])   
print(n2.shape)  # (3, 4) ，说明是2维的，4*3的矩阵

n3 = np.array(['0',9.18,20])
print(n3)  # 结果统一成str了
print(n3.shape)  # (3,)，说明是一维矩阵
n4 = np.array([1,2,3.14])
print(n4)  # 结果统一成float了

# 2.2创建1矩阵和0矩阵
one = np.ones((1,8,3),dtype=int)
one.shape # (1, 8, 3)  # 表示1个矩阵，是8*3的矩阵
print(one)

one = np.ones((2,6,3),dtype=int)
one.shape  # (2, 6, 3) 表示2个矩阵，每个均是6*3的矩阵
print(one)

zero = np.zeros(shape=(100,200,5))  # 创建0矩阵
zero.shape  # ans=(100, 200, 5)

zero = np.zeros(shape=(3,6))  # 创建0矩阵
zero.shape  # (3, 6)

# 2.3 full方法创建矩阵，即填充数据的方法
np.full((4,5),fill_value=8.88)  # 创建一个4*5的矩阵，值均是8.88

# 2.4创建对角矩阵
np.eye(5)  # 生成5阶对角矩阵



'''3.切割均分数据np.linspace,np.arange'''
# 3.1 np.linspace的应用（切割均分数据）
np.linspace(0,10,9,dtype=float)  # 从0到10分成9份
np.linspace(0,100,51,dtype=float)  # 从0到100分成51份
np.linspace(start = 0,stop = 100,num = 50,endpoint=False)  # endpoint=False表示不包含最后一个数据，即150不会包含进去
np.linspace(start = 0,stop = 100,num = 50,endpoint=False,retstep=True)  # retstep=True会显示步长
np.linspace(start = 0,stop = 100,num = 50,endpoint=False,retstep=True,dtype=np.int8)  # dtype=np.int8表示调整类型至np.int8
np.linspace(start = 0,stop = 150,num = 50,endpoint=False,retstep=True,dtype=np.int8)  # int型只能表示到127（0-127），故会出现负数

# 3.2 np.arange的应用（等增长数列）
np.arange(0,100,step = 3)  # 从0开始到100，步长为3，绝对不会包含最后一个数字100
np.arange(0,100,step = 2)  # 从0开始到100，步长为2，绝对不会包含最后一个数字100
np.arange(0,100)  # step不写默认为1
np.arange(100)  # 第一个数不写默认为0



'''4.生成随机矩阵'''
import numpy as np
import matplotlib.pyplot as plt
# 4.0 np.random.rand()生成[0,1)之间随机浮点数  
np.random.rand() # 生成[0,1)之间随机浮点数  
test = np.arange(-100,100,0.01,dtype='int').reshape((20000,1))  # ！生成从-100到100,20000个数据,属性为int类型

# 4.1 np.random.randint生成整数型的数组
np.random.randint(-100,100,size = (4,5))  # 生成4*5的整数矩阵，其数值在-100到100挑选，包含-100但不包含100

# 4.2 np.random.randn生成标准正太分布
np.random.randn(6,6)  # 6*6阶矩阵

# 4.3 np.random.normal生成正太分布
np.random.normal(175,6,6)  # 生成一个均值为175，方差为100的正太分布；共6个数，6*1
n1 = np.random.normal(loc=175,scale=6,size = 6)  # 生成一个均值为175，方差为100的正太分布；共6个数，6*1，与上面完全一样
np.random.normal(175,100,size = (6))  # 生成一个均值为175，方差为100的正太分布；与上面的完全相同
np.random.normal(175,100,size = (3,2))  # 生成一个均值为175，方差为100的正太分布，维度是3*2的矩阵

# 4.4np.random.random生成0到1的数随机数，左闭右开
np.random.random(3)
np.random.random(size = (3))
np.random.random(size = (2,3))  # 生成一个2*3的矩阵
np.random.random(size = (2,3,3))  # 生成二个3*3的矩阵
img = np.random.random(size = (456, 730, 3))
plt.imshow(img)  # 画出图片
plt.show()
plt.imshow(cat)  # 画出图片
plt.show()
print(cat.shape,img.shape)  # 两张图片数据结构相同，仅仅数值不同差距就如此巨大

img = np.random.random(size = (456, 730, 3))
img = img.astype('uint8')  # 元素格式转换
plt.imshow(img) 

# 4.5 np.arange创建矩阵
np.arange(3)

# 4.6查看img的属性
img.ndim   # 维度
img.shape  # 形状
img.size   # 总长度
img.dtype  # 元素类型



import numpy as np
'''5 np.array的基本操作'''
# 5.1 索引，读取np.array
n1 = np.array([1,2,4,7,9])
n1[3]  # 读取第四个数据，答案当然是7
n2 = np.random.random(size = (2,3,3))  # 生成二个3*3的矩阵
n2[0]      # 读取第一个数组
n2[0,1]    # 读取第一个数组，其行为第二行
n2[0,1,1]  # 读取第一个数组，行和列数均为2  （看来和list非常类似）
n2[0:1]    # 读取第一个数组，关键是逗号表示断点
n2[:1]     # 同样是读取第一个数组

# 5.2 切片
n = np.random.randint(0,100,size = 10)  # 生成1*10的整数矩阵，其数值在0到100挑选，包含0但不包含100
n[3:6]  # 读取n的第4个到第7个，但第7个不读，左闭右开；也就是第4个数到第6个数
n[-2:]  # 读取n的倒数第二个到结尾

# 5.2切片
n = np.array([[1,2,3],[4,5,6],[7,8,9]])
n[1]
n[1,[0,2]] # 取第二行的第1列和第3列
n[[0,2]]
n[1,2]
n[[1,2],[2,0]]
n[::-1]
n[:,0:2]
n[1:2,0:2]
n.reshape((9,1))
n.resize((9,1))

# 5.3数据反转
n[::]    # 同n[:]，n不会改变
n[::-1]  # n会数据反转（n会倒着写）
n[::2]   # n会每隔2个数取1个数
n[::-2]  # n会每隔2个数取1个数，不过n会倒着写

# 5.4数据变形reshape，和astype改变数据类型
n = np.random.randint(0,100,size = 64)
n_reshpe = n.reshape(4,4,4)  # 换成4个数组，每个数组为4*4阶矩阵
n_reshpe_2 = n.reshape(8,8)  # 换成8*8阶矩阵
n_reshpe_3 = n_reshpe_2.astype(float)  # 转换数据类型



'''6.级联 np.concatenate() 级联需要注意的点：
1.级联的参数是列表：一定要加中括号或小括号
2.维度必须相同  形状相符
3.级联的方向默认是shape这个tuple的第一个值所代表的维度方向   【重点】
4.可通过axis参数改变级联的方向'''
import numpy as np
# 6.1二维级联
n1 = np.random.randint(0,100,size = (3,3))
n2 = np.random.randint(-100,0,size = (3,3))
np.concatenate((n1,n2),axis = 1)  # axis = 1是列滚动得到3*6矩阵
np.concatenate((n1,n2),axis = 0)  # axis = 0是行滚动得到6*3矩阵
np.c_[n1,n2]  # 这也是一种级联操作
n1.ndim   # 维度
n1.shape  # 形状（各维度的长度）
n1.size   # 总长度
n1.dtype  # 元素类型
n1.itemsize  # 元素每项的大小
n1.data  # 输出数据
n1 = n1.astype(float)  # 更改元素类型
n1.dtype  # 元素类型
n1 = n1.astype('uint8')  # 更改元素类型
n1.dtype  # 元素类型

# 6.2三维级联
import matplotlib.pyplot as plt  # 导入画图
cat = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/cat.jpg') 
img = np.random.random(size = (456, 730, 3))
cat.shape    # (456, 730, 3)
img.shape    # (456, 730, 3)
image = np.concatenate((cat,img),axis = 0)
image.shape  # (912, 730, 3) 
plt.imshow(image)  # 画出图片
image = np.concatenate((cat,img),axis = 1)
image.shape  # (456, 1460, 3)
plt.imshow(image)  # 画出图片

nd1 = np.random.randint(0,10,size=(1,2,3))
nd2 = np.random.randint(0,10,size=(1,4,3))
np.concatenate([nd1,nd2],axis=1)
'''级联法则：
数组级联的时候用axis去指定级联方向，axis默认是0
两个数组抛开参与级联的那个维度以后，剩下的维度如果相同则可以进行级联，否则不能级联
两个数组级联维度必须一致'''

# 6.3np.hstack与np.vstack
# 水平级联与垂直级联，处理自己，进行维度的变更
n = np.random.randint(0,100,size = (4,5))
n1 = np.hstack(n)  # 水平级联
n1.shape     # 得到20*1的矩阵
np.vstack(n) # 垂直级联，在这里会得到原始数据

n = np.random.randint(0,10,size = 5)
np.vstack(n)  # 垂直级联，将水平的数据转换成垂直的



'''7. 切分'''
# 7.1.1 np.split切分
import numpy as np
import matplotlib.pyplot as plt 
cat = plt.imread('G:/anaconda/Spyder 项目/其他/图片包/cat.jpg')
n = np.random.randint(0,100,size = (4,6))
np.split(n,4)  # 按行切分成4份，当然若分成3份则会报错

# 7.1.2 np.split切分图片(三维矩阵)
cat_1 = np.split(cat,3)  # 将cat平均切分为3份，返回列表
cat.shape  # (456, 730, 3)
cat_1[0].shape  # (152, 730, 3)
plt.imshow(cat_1[0])  # 画出图片
plt.imshow(cat_1[1])  # 画出图片
plt.imshow(cat_1[2])  # 画出图片

# 7.1.3 np.split切分图片
cat_2 = np.split(cat,2,axis=1)   # 将cat切分为3份，返回列表
cat_2[0].shape  # (456, 365, 3)
plt.imshow(cat_2[0])  # 画出图片
plt.imshow(cat_2[0])  # 画出图片

# 7.2 np.vsplit水平切分
n = np.random.randint(0,100,size = (3,6))
np.vsplit(n,(2))  # 分成2组，即平分
np.vsplit(n,(1,2))  # 表示0到1分1组，1到2分1组，剩下的1组
np.vsplit(n,(0,1,2))  # 同上面

# 7.3 np.hsplit(n,(2,4))垂直切分
np.hsplit(n,(3))  # 均分成3组
np.hsplit(n,(2,4))  # 编号0到2分1组，2到4分1组，剩下的1组

# 7.4 拷贝
l = [1,2,3,4]
nd = np.array(l)
l[0] = 1000
print(nd)  # [1 2 3 4]，没有变化



