
# -*- coding: utf-8 -*-
"""
@author: 肖
"""
'''概念：
序列化将对象的状态信息转换为可以存储或者传输的形式的过程
而反序列化刚好相反,是指将流转换为对象'''
import matplotlib.pyplot as plt
import pickle as pickle  # pickle导入序列化
import numpy as np

'''1.读取数据'''
label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 图片对应的结果

def load_data(path,index):   # 读取训练文件函数
    # data_batch_1
    # batch:一批，10000个数据
    with open(path+str(index),'rb') as file:
        data = pickle.load(file,encoding='ISO-8859-1')  # 将该二进制的file进行反序列化，'ISO-8859-1'是欧盟的编码格式
        x_data = data['data'].reshape((10000,3,32,32)).transpose((0,2,3,1))
        y_target = data['labels']
        return (x_data,y_target)
    
def load_test_data(path):    # 读取测试文件函数
    #data_batch_1
    #batch:一批，10000个数据
    with open(path,'rb') as file:
        #将该二进制的file进行反序列化
        data = pickle.load(file,encoding='ISO-8859-1')
        x_data = data['data'].reshape((10000,3,32,32)).transpose((0,2,3,1))
        y_target = data['labels']
        return (x_data,y_target)    

path = 'D:/数据分析视频/day134/10-kinds-of-pictures/data_batch_'   # 设置文件路径
load_data(path=path,index = 1)   # 进行第一个文件读取,一个文件有10000张图片
x_data,y_target = load_data(path = path,index = 1)  # 将刚才的数据接收一下，读取第一个文件
x_data.shape  # 由于有10000张图片因此x_data是4维的数组

plt.imshow(x_data[0])    # 画出10000张图片中的第一张图片
label_name[y_target[0]]  # y_target[0]即图片对应的结果，现在将其转换为英文单词

# 在一张图中画出前100张图片
plt.figure(figsize=(12,18))
for i in range(100):
    axes = plt.subplot(10,10,i+1)  # 当i=0时,表示10*10的图片,先画第一张     
    axes.imshow(x_data[i])         # 画出第一张   
    axes.set_title(label_name[y_target[i]])   # 设置标题
    plt.axis('off')

for i in range(1,6):   # 读取全部的5个包,共50000张图片,并保存
    if i == 1:
        x_data,y_target = load_data(path=path,index = i)  # 进行文件的读取
    else: 
        data = load_data(path = path,index = i)
        x_data = np.concatenate([x_data,data[0]])
        y_target = np.concatenate([y_target,data[1]])

x_test,y_test = load_test_data('D:/数据分析视频/day134/10-kinds-of-pictures/test_batch')  # 测试数据！

'''2.使用支持向量机作为机器学习模型'''
from sklearn.svm import SVC
svc = SVC()
X_train = x_data.reshape((50000,-1))   # 对x_data进行数据格式的转换
X_train.shape   # (50000, 3072)，所以需要降维，缩短机器学习训练时间

# 1.PCA降维：
from sklearn.decomposition import PCA   # 使用PCA对X_train降维
pca = PCA(n_components=80,whiten=True)  # 表示最后保持80个主成份

pca.fit(X_train)   # 进行训练
X_train_pca = pca.transform(X_train)  # 进行数据的转换
X_train_pca.shape  # (50000, 80),可以看出数据量减少了
# 2.训练
svc.fit(X_train_pca,y_target)  # 进行svm训练
# 3.对测试数据进行PCA降维
x_test2 = x_test.reshape((10000,-1))   # 对x_test进行数据格式的转换
x_test2_pca = pca.transform(x_test2)   # 对x_test2进行PCA降维
# 4.查看预测结果
y_ = svc.predict(x_test2_pca)   # 对比y_test测试结果

# 5.展示测试结果
plt.figure(figsize = (16,18))
for i in range(100):  
    axes = plt.subplot(10,10,i+1)   
    axes.imshow(x_test[i])  
    pre = label_name[y_[i]]  
    tru = label_name[y_test[i]]   
    axes.set_title('predict:%s \n true: %s'%(pre,tru))   
    plt.axis('off')


