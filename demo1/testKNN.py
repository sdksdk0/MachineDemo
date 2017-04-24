from  numpy import *

from demo1 import KNN

#生成数据集和类别标签
dataSet,lables= KNN.createDataSet()

#定义一个未知类别的数据
testX=array([1.2,1.0])
k=3

#调用分类函数对未知数据分类
outputLabel= KNN.KNNClassify(testX, dataSet, lables, k)

print("你输入的数据是：",testX," 类别为：",outputLabel)


#定义一个未知类别的数据
testX=array([0.7,0.3])


#调用分类函数对未知数据分类
outputLabel= KNN.KNNClassify(testX, dataSet, lables, k)
print("你输入的数据是：",testX," 类别为：",outputLabel)