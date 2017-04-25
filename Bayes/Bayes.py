from  numpy   import *
#过滤网站恶意留言
def loadDatSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        #创建两个集合的冰并集
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

#将文档词条转换成词向量
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else:
            print('这不是一个词汇,%word')
    return returnVec


#朴素贝叶斯分类器训练函数，从词向量计算概率
def trainNBO(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)

    p0Num=ones(numWords);  #避免一个概率值为0,最后的乘积也为0
    p1Num=ones(numWords)
    p0Denom=2.0   #用于统计0类中的总数
    p1Denom=2.0   #用于统计1类中的总数
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)

    if p1>p0:
        return 1
    else:
        return 0

def testingBN():
    listOPosts,listClasses=loadDatSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNBO(array(trainMat),array(listClasses))
    testEntry=['dog','my','stupid']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print('testEntry, classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('testEntry, classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
