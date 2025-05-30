from enum import unique
from math import log
import operator


def calcShannonEnt(dataSet):
    """计算香农熵（Shannon Entropy）
    dataSet:数据集样本
    """
    nmuEntries = len(dataSet)                                     # 获取数据集中样本的总数
    labelCounts = {}
    for featVec in dataSet:                                       # 遍历数据集中的每一行（样本）
        currentLabel = featVec[-1]                                # 当前样本的标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:                                       # 计算该数据集的香农熵
        prob = float(labelCounts[key])/nmuEntries                 # 概率
        shannonEnt -=prob * log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """按照给定特征划分数据集
    dataSet:待划分数据集
    axis:划分数据集的特征
    value:要筛选的特征值
    """
    retDataSet = []
    for featVec in dataSet:                                # 遍历数据集 dataSet 中的每一行（即每个样本）
        if featVec[axis] == value:                         # 检查当前样本的第 axis 列特征值是否等于 value
            reducedFeatVec = featVec[:axis]                # 使用切片操作 featVec[:axis]，获取当前样本中第 axis 列之前的所有特征值
            reducedFeatVec.extend(featVec[axis+1:])        # 使用切片操作 featVec[axis+1:]，获取当前样本中第 axis 列之后的所有特征值；使用 extend 方法将这些特征值添加到 reducedFeatVec 中
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """选择最好的数据集划分方式
    dataSet:待划分数据集
    """
    numFeatures = len(dataSet[0]) - 1                     # 数据集样本中的特征数量
    baseEntropy = calcShannonEnt(dataSet)                 # 计算原始数据集的熵（基线熵）
    bestInfoGain = 0.0;                                   # 初始化信息增益
    bestFeature = -1                                      # 最佳特征索引
    for i in range(numFeatures):                          # 遍历所有特征
        featList = [example[i] for example in dataSet]      # 提取第i列的特征值并放入到 featlist 列表中
        uniqueVals = set(featList)                          # 将列表转化为集合(Set)
        newEntropy = 0.0                                    # 初始化新的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)    # 划分数据集
            prob = len(subDataSet)/float(len(dataSet))      # 计算子数据集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) # 计算子数据集的熵
        infoGain = baseEntropy - newEntropy               # 信息增益：原始熵 - 划分后的熵
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain                       # 如果当前特征的信息增益大于之前的最大信息增益，则更新最佳信息增益和最佳特征索引
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """从类别标签列表中找出出现次数最多的类别
    classList: 一个包含类别标签的列表
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """构建决策树"""
    classList = [example[-1] for example in dataSet]     # 提取数据集中每个样本的类别标签（最后一列）
    if classList.count(classList[0]) == len(classList):  # 检查是否所有类别标签相同，若相同则返回该类别标签作为叶子节点
        return classList[0]
    if len(dataSet[0]) == 1:                      # 检查每个样本中是否只剩下一个值（即类别标签）
        return majorityCnt(classList)             # 调用 majorityCnt 函数，返回 classList 中出现次数最多的类别标签
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 使用 chooseBestFeatureToSplit 函数选择最佳特征（基于信息增益）
    bestFeatLabel = labels[bestFeat]              # 获取最佳特征的标签名称
    myTree = {bestFeatLabel:{}}                   # 以最佳特征标签为键，初始化一个空字典作为决策树的当前节点
    del(labels[bestFeat])                         # 从特征标签列表中删除已选择的最佳特征，避免在子树中重复使用
    featValues = [example[bestFeat] for example in dataSet]      # 提取某个特征的所有值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]             # labels[:] 是对 labels 列表的浅拷贝，确保在递归调用中不会修改原始的 labels 列表
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    """使用决策树进行分类
    inputTree:输入的决策树
    featLabels:特征标签列表
    testVec:待分类的样本
    """
    firstStr = list(inputTree.keys())[0]                      # 获取树的第一个键（特征名称）
    secondDict = inputTree[firstStr]                          # 获取该特征对应的子树
    featIndex = featLabels.index(firstStr)                    # 获取该特征在特征标签列表中的索引
    for key in secondDict.keys():                              # 遍历子树中的每个键（特征值）
        if testVec[featIndex] == key:                          # 如果测试样本在该特征上的值等于当前键，则递归调用 classify 函数
            if isinstance(secondDict[key], dict):              # 如果该键对应的值是一个字典，则递归调用 classify 函数
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]                    # 否则，说明是叶节点，返回该叶节点的类别标签
    return classLabel

def storeTree(inputTree, filename):
    """将决策树存储到文件中
    inputTree:输入的决策树
    filename:文件名
    """
    import pickle
    with open(filename, 'wb') as fw:            # 以二进制写入模式打开文件
        pickle.dump(inputTree, fw)                          # 使用 pickle 模块将决策树对象序列化并写入文件中
        fw.close()                               # 关闭文件

def grabTree(filename):
    """从文件中读取决策树
    filename:文件名
    """
    import pickle
    with open(filename, 'rb') as fr:            # 以二进制读取模式打开文件
        return pickle.load(fr)                            # 使用 pickle 模块从文件中读取决策树对象并返回