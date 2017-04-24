#图像识别
# -*- coding: utf-8 -*-
from  numpy import *
import operator
import os

#分类
def  KNNClassify(newInput,dataSet,lables,k):
    numSamples=dataSet.shape[0]

    diff=tile(newInput,(numSamples,1))-dataSet
    squaredDiff=diff**2
    squaredDist=sum(squaredDiff,axis=1)
    distance=squaredDist**0.5


    #对距离进行排序
    sortedDistances=argsort(distance)
    classCount={}
    for i in range(k):
        voteLable=lables[sortedDistances[i]]
        classCount[voteLable]=classCount.get(voteLable,0)+1

    maxCount=0
    for key,value in classCount.items():
        if value>maxCount:
            maxCount=value
            maxIndex=key

    return maxIndex



#将图片转换为向量
def img2vector(filename):
    rows=32
    cols=32
    imgVector=zeros((1,rows*cols))
    fileIn=open(filename)
    for row in range(rows):
        lineStr=fileIn.readline()
        for col in range(cols):
            imgVector[0,row*32+col]=int(lineStr[col])

    return imgVector


#加载数据集
def loadDataSet():

    print("获取训练数据集")
    dataSetDir='D:/pythonWorkspace/testDemo/'
    trainingFileList=os.listdir(dataSetDir+'trainingDigits')
    numSamples=len(trainingFileList)

    train_x=zeros((numSamples,1024))
    train_y=[]
    for i in range(numSamples):
        filename=trainingFileList[i]

        train_x[i,:]=img2vector(dataSetDir+'trainingDigits/%s'%filename)

        label=int(filename.split('_')[0])
        train_y.append(label)

    print("获取测试数据集")
    testingFileList=os.listdir(dataSetDir+'testDigits')
    numSamples=len(testingFileList)
    test_x=zeros((numSamples,1024))
    test_y=[]
    for i in range(numSamples):
        filename=testingFileList[i]

        test_x[i,:]=img2vector(dataSetDir+'testDigits/%s'%filename)

        label=int(filename.split('_')[0])
        test_y.append(label)
    print(test_y)
    return train_x,train_y,test_x,test_y


#手写识别主流程
def testHandWritingClass():
    print("加载数据")
    train_x, train_y, test_x, test_y=loadDataSet()


    print("训练模型")
    pass

    print("测试")
    numTestSamples=test_x.shape[0]
    matchCount=0
    for i in range(numTestSamples):
        predict=KNNClassify(test_x[i],train_x,train_y,3)
        if predict==test_y[i]:
            matchCount+=1
        accuracy=float(matchCount)/numTestSamples

    print("输出结果")
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))











