from  numpy   import *
#过滤垃圾邮件
import Bayes

def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[];classList=[];fullText=[]
    for  i in  range(1,26):
        wordList=textParse(open('testDemo/email/span/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('testDemo/email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=Bayes.createVocabList(docList)
    trainingSet=range(50);
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        trainMat=[];trainClasses=[];
        for docIndex in trainingSet:
            trainMat.append(Bayes.setOfWords2Vec(vocabList,docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam=Bayes.trainNBO(array(trainMat),array(trainClasses))
        errorCount=0
        for docIndex in testSet:
            wordVector = Bayes.setOfWords2Vec(vocabList, docList[docIndex])
            if  Bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
        print('the error rate is: ', float(errorCount) / len(testSet))

spamTest()