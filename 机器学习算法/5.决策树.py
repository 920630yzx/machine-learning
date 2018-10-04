# -*- coding: utf-8 -*-
"""
@author: 肖
决策树（decision tree）是一个树结构（可以是二叉树或非二叉树）。
其每个非叶节点表示一个特征属性上的测试，每个分支代表这个特征属性在某个值域上的输出，而每个叶节点存放一个类别。
使用决策树进行决策的过程就是从根节点开始，测试待分类项中相应的特征属性，并按照其值选择输出分支，直到到达叶子节点，将叶子节点存放的类别作为决策结果。
K-近邻算法可以完成很多分类任务，但是它最大的缺点就是无法给出数据的内在含义，决策树的主要优势就在于数据形式非常容易理解。
决策树算法能够读取数据集合，构建类似于上面的决策树。决策树很多任务都是为了数据中所蕴含的知识信息，
因此决策树可以使用不熟悉的数据集合，并从中提取出一系列规则，机器学习算法最终将使用这些机器从数据集中创造的规则。
专家系统中经常使用决策树，而且决策树给出结果往往可以匹敌在当前领域具有几十年工作经验的人类专家。

决策树优缺点：
优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。既能用于分类，也能用于回归
缺点：可能会产生过度匹配问题
"""

from sklearn.tree import DecisionTreeClassifier
import sklearn.datasets as datasets   # 导入数据

'''1.使用自带的iris数据集'''
iris = datasets.load_iris()
x_data = iris.data  # 获取数据,每行数据由4个属性构成
y_target = iris.target  # 获取数据对应的结果,分成3类

from sklearn.model_selection import train_test_split   # 导入分类的包
x_train,x_test,y_train,y_test = train_test_split(x_data,y_target,test_size = 0.1)
# x_train,y_train表示训练数据及结果； x_test,y_test表示测试数据及结果

'''2.使用决策树算法---这里写了3种'''
tree = DecisionTreeClassifier(max_depth=5)  # 使用决策树算法,max_depth=5是树的最大深度(其实这里只能到4，也就是说max_depth给5或者给6或者给4都是一样的)
# 如果max_depth不进行声明,那么,有多少个属性,树的深度就是多少;这里x_data有4个属性,因此默认max_depth=4；
# 如果属性太多，此时就需要限定树的深度
# 比如有200个属性，max_depth会根据信息增益的方式进行选择，将最重要的100个属性选出，然后进行数据的分类
tree.fit(x_train,y_train)  # 进行训练
tree.predict(x_test)  # 输出预测结果进行对比
tree.score(x_test,y_test)  # 查看得分结果

tree = DecisionTreeClassifier(max_depth=2)  # 设置max_depth=2进行机器学习
tree.fit(x_train,y_train)
tree.score(x_test,y_test)

tree = DecisionTreeClassifier(max_depth=5)  # 设置max_depth=2进行机器学习
tree.fit(x_data,y_target)  # 对所有的数据进行训练
tree.score(x_data,y_target)  # 进行打分
tree.score(x_test,y_test)    # 进行打分

'''3.使用KNN算法'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(5)  # n_neighbors=5,表示邻居的个数
knn.fit(x_data,y_target).score(x_data,y_target)

'''4.使用逻辑斯蒂回归算法
# 决策树：信息论 信息熵将划分展示
# 逻辑斯底：概率论 '''
from sklearn.linear_model import LogisticRegression
lrg = LogisticRegression()
lrg.fit(x_data,y_target).score(x_data,y_target)

'''5.使用决策树回归预测一个椭圆'''
from sklearn.tree import DecisionTreeRegressor
import numpy as np
x_data = np.sort(200*np.random.rand(100,1) - 100,axis = 0)  # 随机生成-100到100的数据,再使用np.sort方法进行排序
x = np.array(x_data)
np.argsort(x)

dot_x = np.pi*np.sin(x_data)
dot_y = np.pi*np.cos(x_data)
y_target = np.c_[dot_x,dot_y]  # 级联
y_target[::5] += np.random.randn(20,2)*0.2  # 添加一些噪音
import matplotlib.pyplot as plt
plt.scatter(y_target[:,0],y_target[:,1])

# 5.1 现在使用决策树来进行回归---创建了3个不同深度的决策树
tree_2 = DecisionTreeRegressor(max_depth=2)
tree_5 = DecisionTreeRegressor(max_depth=5)
tree_20 = DecisionTreeRegressor(max_depth=20)

tree_2.fit(x_data,y_target)
tree_5.fit(x_data,y_target)
tree_20.fit(x_data,y_target)

# 5.2 创造预测数据并画图-点图
x_test = np.arange(-100,100,0.01).reshape((20000,1))
y2_ = tree_2.predict(x_test)
plt.scatter(y2_[:,0],y2_[:,1])  # y2_[:,0]是y2_第一列,点越少说明种类也越少

y5_ = tree_5.predict(x_test)
plt.scatter(y5_[:,0],y5_[:,1])

y20_ = tree_20.predict(x_test)
plt.scatter(y20_[:,0],y20_[:,1])



