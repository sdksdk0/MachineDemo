from  numpy import *
#加载数据
def  loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for  line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=map(float,curLine)   #变成float类型
        dataMat.append(fltLine)
    return dataMat

#计算欧几里得距离
def  distEcud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#构建聚簇中心，取k个随机质心
def randCent(dataSet,k):
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))   #每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ=min(dataSet[:,j])
        maxJ=max(dataSet[:,j])
        rangeJ=float(maxJ-minJ)
        centroids[:,j]=minJ+rangeJ * random.rand(k,1)
    return centroids


#k-means聚类算法

def kMeans(dataSet,k,distMeans=distEcud,createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))  #用于存放该样本属于哪类及质心距离
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf;minIndex=-1
            for j in range(k):
                distJl=distMeans(centroids[j,:],dataSet[i,:])
                if distJl<minDist:
                    minDist=distJl;minIndex=j
            if clusterAssment[i,0] != minIndex:clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist ** 2
        print(centroids)
        for cent in range(k):
            #去第一列等于cent的所有列
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
        return centroids,clusterAssment
















