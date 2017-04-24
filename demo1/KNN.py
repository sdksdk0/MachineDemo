#KNN
from numpy  import*
import operator

#创建一个数据集，包含两个类别共4个样本
def createDataSet():

    #生成一个矩阵，每行表示一个样本
    group = array([[1.0,0.9],[1.0,1.0],[0.1,0.2],[0.0,0.1]])
    #4个样本分别所属的类别
    labels=['A','A','B','B']
    return group,labels

#KNN分类算法函数定义
# 输入:      newInput:  (1xN)的待分类向量
#             dataSet:   (NxM)的训练数据集
#             labels: 	训练数据集的类别标签向量
#             k: 		近邻数
def KNNClassify(newInput,dataSet,labels,k):
    numSamples=dataSet.shape[0]

    #1、计算距离
    diff=tile(newInput,(numSamples,1))-dataSet   #元素差值
    squaredDiff=diff**2    #将差值平方
    squaredDist=sum(squaredDiff,axis=1)  #按行累加
    distance=squaredDist**0.5   #将差值平方和进行开方，就是距离


    #2、对距离排序
    sortedDistIndices=argsort(distance)
    classCount={}
    for i in range(k):

        # 选择k个最近邻
        voteLabel=labels[sortedDistIndices[i]]

        #4、计算k个最近邻中各类别出现的次数
        classCount[voteLabel]=classCount.get(voteLabel,0)+1

    #5、返回出现次数最多的类别标签
    maxCount=0

    for key,value in classCount.items():
        if value>maxCount:
            maxCount=value
            maxIndex=key

    return maxIndex







