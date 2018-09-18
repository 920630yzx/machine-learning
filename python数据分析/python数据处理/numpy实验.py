import pandas as pd
import numpy as np
np.__version__ # 查看numpy版本


#1.定义矩阵(数组)  矩阵相乘及转置
A=np.array([[1,2],
            [4,5],
            [7,8]])
B=np.array([[0,1,2],[3,4,5]])
C=np.dot(A,B)  # 求矩阵之积
print(C)   # 打印矩阵
print(C.T) # 输出矩阵C的转置矩阵


#2.1定义随机矩阵(数组),reshape定义矩阵的行数和列数
D1=np.arange(9)
D2=np.arange(9).reshape(3,3)
D3=np.arange(27).reshape(3,3,3)

# 2.2定义随机矩阵
#生成2*3的随机矩阵 (元素在（0，1）之间):
np.random.rand(2, 3) 
#生成3*4的矩阵,元素在0-10之间,不包含0和10
np.random.randint(0, 10, size=(3,4))
#生成3个随机数:
np.random.random(3)
#生成10正太分布数:
np.random.normal(size=10)


#3.1读取矩阵(数组) 
print(D3[:,1])  # 所有的矩阵的第二行
print(D3[:,1,0]) # 所有的矩阵的第二行第一列
print(D3[1:])
print(D3[1,:])
print(D3[1,1,:])
print(D3[1])  # 第二个矩阵
print(D3[1][1])  # 第二个矩阵的第二行
print(D3[1][1][1])    # 第二个矩阵的第二行第二列

#3.2迭代读取,全部读取
b = np.arange(27).reshape(3, 3, 3)
for krow1 in b[0][0]:
    print(krow1)

print(list(b.flat)) # b.flat,读取b中全部数据
for element in b.flat: # 循环读取
    print(element)


#4.创建0数组,1数组,和单位矩阵
print(np.zeros((3,3,3)))
print(np.ones((3,3,3)))
print(np.eye(5)) # 生成5阶单位矩阵


# 5.查看数组类型,转换数组中元素的类型
b = np.arange(27).reshape(3, 3, 3)
print(b)
print(b.ndim) #数组轴的个数
print(b.size) # 数组元素的总个数
print(b.dtype)  # 一个用来描述数组中元素类型的对象

c = np.array([1,2,3,4,5])
print(c.dtype)  # 查看元素类型
d = np.array([1,2,3,4,5], dtype='float64') # 转换数组中元素的类型!!!
print(d.dtype)  # 查看元素类型


#6.均分函数np.linspace
g = np.linspace(0,np.pi,3)  # 将0到Pi均分为3份


#7.1数组基本运算
import pandas as pd
import numpy as np
e = np.array([20,30,40,50])
f = np.arange(4)
print(e-f) #求差
print(e*2) #求积
g = np.linspace(0,np.pi,3) # 返回输入数量的等比间隔，linspace(start, stop, num=50)
print(g.sum()) #求和
print(g.cumsum())  # 累计求和
print(np.log(g))  # 求对数,对数组中每一个元素都求其对数
print(np.exp(g))  # 求指数,对数组中每一个元素都求其指数
print(np.sqrt(g)) # 求开方,对数组中每一个元素都求其开发
C=np.dot(A,B)  #求矩阵之积,A、B之前已定义;#求矩阵之积

#7.2求逆矩阵np.linalg.inv()
X=np.array([[2,1],[1,-2]])
X_=np.linalg.inv(X)  # 求矩阵X的逆矩阵
print(X)
print(X_)
print(np.dot(X,X_))

# 7.3 求矩阵的秩
np.linalg.matrix_rank(X) 

# 7.4求矩阵解线代  题目：
# 2x+y=1
# x-2y=1 则：
b1 = np.array([[1],[1]])  # 线代中的b值
X = np.array([[2,1],[1,-2]]) # 线代中的A值
X_= np.linalg.inv(X)
z = np.dot(X_,b1)  # 求矩阵解线代
print(z)


# 8.更改数组的形状
# 8.1将多维数组降位一维a.ravel()
a = np.arange(9).reshape(3, 3)
print(a)
print(a.ravel()) # 将多维数组降位一维

#8.2转变数组形状resize,改变数组行数和列数
a1.resize(2,6) #转变数组形状,这会改变原有数组
print(a1)

#8.3横向添加数组np.vstack
a = np.arange(9).reshape(3, 3)
a1 = np.array([9,10,11]) 
a2 = np.vstack((a,a1)) # 横向添加数组，不改变原有数组
print(a2)


#8.4纵向添加数据np.c_
a3 = np.c_[a,a1]
print(a3)


#8.5更好的添加数据方式insert
a = np.arange(9).reshape(3, 3)
a1 = np.array([9,10,11]) 
a4 = np.insert(a, 1, values=a1, axis=0)  # 对矩阵a进行行添加(axis=0),位置在第二行之前
print(a4)
a5 = np.insert(a, 1, values=a1, axis=1)  # 对矩阵a进行列添加(axis=1),位置在第二列之前
print(a5)











