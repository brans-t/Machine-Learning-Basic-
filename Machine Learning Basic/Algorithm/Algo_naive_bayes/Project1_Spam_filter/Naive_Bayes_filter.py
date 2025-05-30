import random

from numpy import *

def loadDataSet():
    """定义一个帖子列表，每个列表是一个帖子，每个帖子是一个单词列表"""
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]                   # 定义每个帖子的类别标签，0和1代表不同的类别
    return postingList, classVec

def createVocabList(dataSet):
    """创建词汇表
    dataSet : 输入文档列表
    """
    vocabSet = set([])                              # 初始化一个空集合，用于存储所有唯一的单词
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """将单词集合转换为向量 --- set of words model
    vocabList : 词汇表
    inputSet : 输入文档
    该函数输出文档向量，向量的每一元素分别为 0 或 1,分别表示词汇表（vocabList）中的单词是否在输入文档（inputSet）中出现
    """
    returnVec = [0] * len(vocabList)                    # 初始化一个全零向量，长度等于词汇表的长度
    for word in inputSet:                               # 遍历输入集合中的每个单词
        if word in vocabList:                           # 如果单词在词汇表中
            returnVec[vocabList.index(word)] = 1        # 在向量中找到单词的索引位置，并将其设置为1
        else: print(f"the world: {word} is not in my Vocabulary!")
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    """
    将输入的文本集合转换为词袋模型向量。

    参数:
    vocabList (list): 词汇表，包含所有可能的单词。
    inputSet (list): 输入的文本集合，包含一组单词。

    返回:
    returnVec (list): 词袋模型向量，表示输入集中每个单词在词汇表中的出现次数。
    """
    returnVec = [0] * len(vocabList)                                      # 初始化一个与词汇表长度相同的向量，所有元素初始值为0
    for word in inputSet:                                                 # 遍历输入集中的每个单词
        if word in vocabList:                                             # 如果在词汇表中，找到该单词在词汇表中的索引
            returnVec[vocabList.index(word)] += 1                         # 并将对应位置的向量值加1
    return returnVec                                                      # 返回最终的词袋模型向量

def trainNB0(trainMatrix, trainCategory):
    """

    Naive Bayes 分类器训练函数
    trainMatrix: 训练集的文档向量矩阵，每一行是一个文档的词向量
    trainCategory: 训练集的类别标签向量，每个元素表示对应文档的类别

    """
    numTrainDocs = len(trainMatrix)  # 获取训练文档的数量
    numWords = len(trainMatrix[0])   # 获取词汇表的长度（假设所有文档向量长度一致）
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算类别1（例如垃圾邮件）的概率

    # 初始化每个类别的单词计数向量和分母
    p0Num = ones(numWords)  # 类别0的单词计数向量
    p1Num = ones(numWords)  # 类别1的单词计数向量
    p0Denom = 2.0  # 类别0的单词总数，若需要归一化，最后这里应该是加上词汇表的长度len(p0Num) or sum(p0Num)
    p1Denom = 2.0  # 类别1的单词总数

    # 遍历所有训练文档
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                       # 如果文档属于类别1
            p1Num += trainMatrix[i]                     # 累加类别1的单词出现次数
            p1Denom += sum(trainMatrix[i])              # 累加类别1的单词总数
        else:  # 如果文档属于类别0
            p0Num += trainMatrix[i]                     # 累加类别0的单词出现次数
            p0Denom += sum(trainMatrix[i])              # 累加类别0的单词总数

    # 计算每个类别的条件概率
    p1Vect = log(p1Num / p1Denom)                             # 类别1的条件概率向量
    p0Vect = log(p0Num / p0Denom)                             # 类别0的条件概率向量

    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    使用朴素贝叶斯分类器对文档进行分类。

    Parameters:
    vec2Classify (list or numpy.array): 待分类文档的特征向量，表示词汇表中单词的出现情况。
    p0Vec (list or numpy.array): 类别0的条件概率向量，表示在类别0下每个单词出现的概率。
    p1Vec (list or numpy.array): 类别1的条件概率向量，表示在类别1下每个单词出现的概率。
    pClass1 (float): 类别1的先验概率，即文档属于类别1的概率。

    Returns:
    int: 分类结果，1表示类别1，0表示类别0。

    Description:
    该函数通过计算待分类文档属于类别1和类别0的后验概率，并比较这两个概率的大小来确定文档的类别。
    使用对数概率来避免数值下溢问题，并提高计算的稳定性。
    """

    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    # 计算类别1的后验概率（对数形式）
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)  # 计算类别0的后验概率（对数形式）

    # 比较两个类别的后验概率，返回概率较大的类别
    if p1 > p0:
        return 1  # 文档属于类别1
    else:
        return 0  # 文档属于类别0

def testingNB():
    """
    测试朴素贝叶斯分类器的完整流程。

    Description:
    1. 加载数据集并创建词汇表。
    2. 将训练数据转换为词向量矩阵。
    3. 使用训练数据训练朴素贝叶斯分类器。
    4. 对测试数据进行分类，并打印分类结果。

    Returns:
    None
    """
    # 加载数据集
    listOPosts, listClasses = loadDataSet()  # 获取训练数据和类别标签

    # 创建词汇表
    myVocabList = createVocabList(listOPosts)  # 生成词汇表

    # 将训练数据转换为词向量矩阵
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将每篇文档转换为词向量

    # 训练朴素贝叶斯分类器
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))  # 训练模型，获取条件概率和先验概率

    # 测试数据1
    testEntry = ['love', 'my', 'dalmation']  # 测试文档1
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 将测试文档转换为词向量
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))  # 打印分类结果

    # 测试数据2
    testEntry = ['stupid', 'garbage']  # 测试文档2
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 将测试文档转换为词向量
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))  # 打印分类结果


def textParse(bigString):
    """切分文本"""
    import re
    listOfTokens = re.split(r'\w*', bigString)                                # 将字符串分割为多个子字符串 \w* 表示匹配多个任意字母
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 定义一个函数 spamTest，用于测试垃圾邮件分类器
def spamTest():
    """
    测试垃圾邮件分类器的准确率。
    使用朴素贝叶斯算法对邮件进行分类（垃圾邮件或非垃圾邮件）。
    """
    # 初始化三个列表：
    # docList：存储每封邮件的单词列表
    # classList：存储每封邮件的类别（1表示垃圾邮件，0表示非垃圾邮件）
    # fullText：存储所有邮件的所有单词
    docList = []  # 每封邮件的单词列表
    classList = []  # 每封邮件的类别
    fullText = []  # 所有邮件的单词（用于创建词汇表）

    # 遍历1到25的整数，分别读取垃圾邮件和非垃圾邮件文件
    for i in range(1, 26):
        # 读取垃圾邮件文件（路径为'email/spam/%d.txt'）
        wordList = textParse(open(f'Project1_Spam_filter/Database/email/spam/{i}.txt', encoding='windows-1252').read())
        docList.append(wordList)  # 将垃圾邮件的单词列表添加到docList
        fullText.extend(wordList)  # 将垃圾邮件的单词添加到fullText
        classList.append(1)  # 垃圾邮件的类别标记为1

        # 读取非垃圾邮件文件（路径为'email/ham/%d.txt'）
        wordList = textParse(open(f'Project1_Spam_filter/Database/email/ham/{i}.txt', encoding='windows-1252').read())
        docList.append(wordList)  # 将非垃圾邮件的单词列表添加到docList
        fullText.extend(wordList)  # 将非垃圾邮件的单词添加到fullText
        classList.append(0)  # 非垃圾邮件的类别标记为0

    # 创建词汇表(集合)，包含所有邮件中出现的单词（去重）
    vocabList = createVocabList(docList)

    # 初始化训练集和测试集
    trainingSet = list(range(50))  # 初始训练集包含所有邮件（50封）
    testSet = []  # 测试集为空

    # 随机选择10封邮件作为测试集
    import random
    for i in range(10):
        # random.uniform(a, b) 是 random 模块中的一个函数，用于生成一个在区间 [a, b) 内的随机浮点数。这里的区间是左闭右开的，即包括 a，但不包括 b
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 将随机选择的邮件索引添加到测试集
        testSet.append(trainingSet[randIndex])
        # 从训练集中删除该邮件索引
        del (trainingSet[randIndex])

    # 初始化训练集的矩阵和类别
    trainMat = []  # 存储训练集的词向量矩阵
    trainClasses = []  # 存储训练集的类别

    # 构建训练集的词向量矩阵和类别
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将每封邮件的单词列表转换为词向量（0或1表示单词是否出现）
        trainClasses.append(classList[docIndex])  # 添加对应的类别

    # 使用训练集训练朴素贝叶斯模型
    # 返回非垃圾邮件的概率向量(p0V)、垃圾邮件的概率向量(p1V)和垃圾邮件的先验概率(pSpam)
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    # 初始化错误计数
    errorCount = 0

    # 测试模型的准确率
    for docIndex in testSet:
        # 将测试集邮件的单词列表转换为词向量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 使用分类函数 classifyNB 对测试集邮件进行分类
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 如果分类结果与实际类别不符，错误计数加1
            errorCount += 1

    # 打印错误率
    print(f"The error rate is: {float(errorCount) / len(testSet)}")

spamTest()  # 调用 spamTest 函数进行测试