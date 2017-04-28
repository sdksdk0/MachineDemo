from  numpy   import *
import time
import matplotlib.pyplot as plt


#加载数据
print('加载数据')
dataSet=[]
fileIn=open('D:/pythonWorkspace/testDemo/kmeans/testSet.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split('\t')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])

#聚类
print('聚类')
dataSet=mat(dataSet)
k=4
centroids,clusterAssment=plt.kmeans(dataSet,k)

#显示结果
print('显示结果')
plt.showCluster(dataSet, k, centroids, clusterAssment)

















