def classify0(inX, dataSet, labels, k):
    """kNN algorithm:
    基于训练数据集dataSet和对应的标签labels，对输入的数据集inX进行分类，k为考虑的最近邻的个数。
    inX: 输入数据集
    dataSet: 训练数据集
    labels: 训练数据集对应的标签
    k: 选择的最近邻的个数
    """
    dataSetSize = dataSet.shape[0]                                     #确认训练数据集中的样本数量；shape：这是 NumPy 数组的一个属性，返回一个元组，表示数组的维度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet                    #将输入向量inX复制成与训练数据集相同大小的矩阵，然后再与样本数据相减
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                                #计算欧氏距离；axis=1：沿着行的方向进行操作，即对每一行进行操作
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()                           #对距离进行排序以，并返回数组元素的排序索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]                     #收集前k个最近邻样本标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1     #统计前k个最近邻样本，每个标签出现的次数
    sortedClassCount = sorted(classCount.items(),                      #对标签次数进行降序排列
                            key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]