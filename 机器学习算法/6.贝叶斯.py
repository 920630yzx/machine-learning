# -*- coding: utf-8 -*-
"""
@author: 肖
朴素贝叶斯算法是一个典型的统计学习方法，主要理论基础就是一个贝叶斯公式

优点：
朴素贝叶斯模型发源于古典数学理论，有着坚实的数学基础，以及稳定的分类效率；
对小规模的数据表现很好；
能处理多分类任务，适合增量式训练；
对缺失数据不太敏感，算法也比较简单，常用于文本分类

缺点：
只能用于分类问题
需要计算先验概率；
分类决策存在错误率；
对输入数据的表达形式很敏感
"""

'''1.高斯分布朴素贝叶斯
用途：高斯分布就是正态分布，一般用于解决分类问题'''
from sklearn.naive_bayes import GaussianNB
import sklearn.datasets as datasets  # 获取数据

iris = datasets.load_iris()
x_data = iris.data
y_target = iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_target,test_size=0.1)

g_NB = GaussianNB()
g_NB.fit(x_train,y_train)
g_NB.score(x_test,y_test)

'''2.多项式分布朴素贝叶斯
用途：适用于文本数据（特征表示的是次数，例如某个词语的出现次数'''
from sklearn.naive_bayes import MultinomialNB

m_NB = MultinomialNB()
m_NB.fit(x_train,y_train)
m_NB.score(x_test,y_test)

'''3.伯努利分布朴素贝叶斯
用途：适用于伯努利分布，也适用于文本数据（此时特征表示的是是否出现，例如某个词语的出现为1，不出现为0）
绝大多数情况下表现不如多项式分布，但有的时候伯努利分布表现得要比多项式分布要好，尤其是对于小数量级的文本数据'''
from sklearn.naive_bayes import BernoulliNB
b_NB = BernoulliNB()
b_NB.fit(x_train,y_train)
b_NB.score(x_test,y_test)



'''4.短信文本分类实战'''
from sklearn.naive_bayes import GaussianNB      # 高斯分布朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB   # 多项式分布朴素贝叶斯
from sklearn.naive_bayes import BernoulliNB     # 伯努利分布朴素贝叶斯
import pandas as pd
sms = pd.read_table('G:/python doc/spyder doc/SMSSpamCollection',header=None)
print(sms.shape)
print(sms.head())  # spam表示垃圾短信

x_train = sms[1]   # 训练数据
y_train = sms[0]   # 训练结果

g_NB = GaussianNB()
g_NB.fit(x_train,y_train)  #  报错：could not convert string to float: 'Rofl. Its true to its name'
# 报错理由：x_train是一组字符串的数据(文本数据)，需要转换成数字

'''解决办法:
tf.fit_transform():  注意参数必须是字符串的一维数组（比如列表或者Series）
返回的是一个稀疏矩阵类型的对象，行数为样本数，列数为所有出现的单词统计个数'''
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()  # 声明对象,调用对象方法
tf.fit(x_train)  # 对文本数据进行训练
X_train = tf.transform(x_train)  # 对文本数据进行转换---->得到稀疏矩阵对象
X_train.shape  # (5572, 8713) 表示5572条短信和8713个不同的单词
X_train.toarray()  # 查看具体的数据

'''4.1 使用高斯分布朴素贝叶斯-训练数据，进行机器学习'''
g_NB = GaussianNB()  # 调用高斯分布方法
g_NB.fit(X_train.toarray(),y_train)   # 进行训练，需使用.toarray()方法将 X_train转换成数据形式

# 进行预测,输入一条短信
message = 'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL'
x_test = tf.transform([message])  # 同样需要对文本数据进行转换---->得到稀疏矩阵对象
x_test
x_test.shape
x_test.toarray()
g_NB.predict(x_test.toarray())   # 预测结果,得到['spam']说明是垃圾短信

'''4.2 使用多项式分布朴素贝叶斯-训练数据，进行机器学习'''
m_NB = MultinomialNB()  # 调用多项式方法
m_NB.fit(X_train,y_train)   # X_train.toarray()也可
message = 'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL'
x_test = tf.transform([message]) 
x_test.toarray()
m_NB.predict(x_test.toarray())    # 预测结果,得到['spam']说明是垃圾短信

'''4.3 使用伯努利分布朴素贝叶斯-训练数据，进行机器学习'''
b_NB = BernoulliNB()
b_NB.fit(X_train,y_train)  # X_train.toarray()也可,就是要慢一点
message = 'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL'
x_test = tf.transform([message]) 
x_test.toarray()
m_NB.predict(x_test.toarray())    # 预测结果,得到['spam']说明是垃圾短信









