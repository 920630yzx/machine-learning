# -*- coding: utf-8 -*-
"""
分类问题：from sklearn.neighbors import KNeighborsClassifier
回归问题：from sklearn.neighbors import KNeighborsRegressor
"""
'''1.案例1---KNN分类案例'''
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
X_train = np.array([[175,65,43],[160,50,37],[180,75,44],[175,55,40],[165,65,41]])
y_labels = ['男','女','男','女','男']
knn = KNeighborsClassifier()  # 调用算法
knn.fit(X_train,y_labels)  # X_train的数据与y_labels中的数据一一对应,随着样本量的增大,准确率也会随之增大

# 进行预测
x_test = np.array([[172,68,42]])   # 输入条件
knn.predict(x_test)   # 输出机器预测的结果



'''2.案例2---KNN分类案例'''
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn.datasets as datasets
iris = datasets.load_iris()  # iris是一个字典


target = iris.target   # 获取数据,一共三种类型的花
data = iris.data  # 获取数据,每种花的特点
from sklearn.model_selection import train_test_split  # 导入拆分的工具,这是随机分类的.
# 让计算机学习一部分数据，另一部分，计算机没有接触过，让计算机预测没有接触的数据，分类，识别这些数据
X_train,x_test,y_train,y_test = train_test_split(data,target,test_size = 0.1)  
# 原data,target均会被分成2份;test_size = 0.1说明留下了10%作为测试数据,这是随机分类的.
# X_train表示训练数据,y_train表示训练数据所对应的结果

# 训练数据:
knn = KNeighborsClassifier(n_neighbors= 10)  # 默认n_neighbors=5,表示选择5个数字进行距离计算并得出其平均值(也就是只找最近的5个数据)
knn.fit(X_train,y_train)  # 进行训练
knn.predict(x_test)   # 预测数据
print(y_test)     # 对比真实的结果,应该是大致相同的

knn.score(x_test,y_test)  #　相当于求knn.predict(x_test),y_test测试的得分情况（准确率）
knn.score(data,target)   #　相当于求knn.predict(data),target测试的得分情况

# 绘制训练前的结果:
import matplotlib.pyplot as plt
data1 = data[:,:2]  # 取头两列
plt.scatter([0,1,2],[1,0,-1],c=[1,2,3])  # c=[1,2,3]表示颜色分成3种
plt.scatter([0,1,2],[1,0,-1],c=['r','g','b'])  # c=['r','g','b']设置具体的颜色
plt.scatter(data1[:,0],data1[:,1],c=target,cmap='rainbow')  # 这仅仅是二维,因此这个图其实只能代表部分的情况,肯定是不太准确



'''3.案例2---KNN分类案例---接上例'''
knn = KNeighborsClassifier(10)
knn.fit(data1,target)  # 训练处理之后的数据data

# 提取x，y轴的范围
x_min,x_max = data[:,0].min(),data[:,0].max()
y_min,y_max = data[:,1].min(),data[:,1].max()

x = np.linspace(x_min,x_max,1000)
y = np.linspace(y_min,y_max,650)
X,Y = np.meshgrid(x,y)  # 交叉处理
print(X.shape,Y.shape)  # (650, 1000) (650, 1000)

xy_test = np.c_[X.ravel(),Y.ravel()]  # X.ravel()是对numpy.ndarray进行一维化处理,np.c_
xy_test.shape  # (650000, 2)
y_ = knn.predict(xy_test)  # 进行预测，得到每一个点的预测结果
plt.pcolormesh(X,Y,y_.reshape(650, 1000))  # 显示数据---结果
plt.scatter(data[:,0],data[:,1],c = target,cmap = 'rainbow')  # 显示数据---这只进行比较



'''4.案例3---KNN回归案例---用于回归：回归用于对趋势的预测，找到合适的函数，进行后续预测
---文字识别案例'''
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier  # 默认n_neighbors=5,表示选择5个数字进行距离计算并得出其平均值 
from sklearn.model_selection import train_test_split  #导入拆分的工具,这是随机分类的.
import matplotlib.pyplot as plt

X_train = np.linspace(0,10*np.pi,200).reshape((200,1))
y_train = np.sin(X_train)

# 读取bmp文件！
file = open('G:/python doc/machine learning/data/0/0_1.bmp',mode = 'rb')
file.read()

data = []
target = []
def read_icon(index): #读取数据
    for i in range(1,501):
        digits = plt.imread('G:/python doc/machine learning/data/%d/%d_%d.bmp'%(index,index,i))  # 拼接法读取
        data.append(digits)   # data是数据
        target.append(index)  # 数据对应的目标，标签，labels

for i in range(10):  # 循环读取
    read_icon(i)
    
plt.imshow(data[0],cmap='gray')  # 画一画第一张图

# 将data和target升级到numpy
data_1 = np.array(data)
target_1 = np.array(target)
print(data.shape)
print(target.shape)

x_data = data_1.reshape((5000,-1))  # knn行机器学习的运算必须转换成二维才能使用，target_1本身就是二维的对应于x_data的每一行
x_data.shape

X_train,x_test,y_train,y_test = train_test_split(x_data,target,test_size = 0.05)  # 留下5%的数据进行测试，其余用于训练，分割是随机的
# X_train表示训练数据,y_train表示训练数据所对应的结果,x_test表示测试数据,y_test表示测试对应的结果

knn = KNeighborsClassifier(20)  # 默认n_neighbors=5,表示选择5个数字进行距离计算并得出其平均值(也就是只找最近的5个数据)
knn.fit(X_train,y_train)  # 进行训练
y_ = knn.predict(x_test)  # 进行预测,可与y_test进行对比
print(y_test)  # 进行对比
knn.score(x_test,y_test)  # 得到预测结果的分数,knn.predict(x_test)与y_test对比的准确率 

# 画出结果：  （当然也不一定需要做这步,通过比较y_与y_test的差异可以看出结果）
plt.figure(figsize=(3*2.4,5*3))  # 5行5列,每个小图片是2.4*3的大小

for i in range(1,26):   
    axes = plt.subplot(5,5,i)       
    image = x_test[(i-1)*10].reshape((28,28))  # 图片的数据x_test
    axes.imshow(image)    
    axes.set_title('true:%d\n predict:%d'%(y_test[(i-1)*10],y_[(i-1)*10]))












