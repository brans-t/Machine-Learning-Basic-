import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")                # 定义决策节点样式
leafNode = dict(boxstyle = "round4", fc = "0.8")                      # 定义叶节点样式
arrow_args = dict(arrowstyle = "<-")                                  # 定义箭头的样式；值是 "<-"：这意味着箭头的样式将是一个简单的线条，箭头指向注释文本的方向

def plotMidText(cntrPt, parentPt, txtString):
    """在父节点和子节点之间绘制文本
    cntrPt: 子节点的中心位置，使用轴的分数坐标表示
    parentPt: 父节点的位置，用于确定箭头的方向，使用轴的分数坐标表示
    txtString: 要显示的文本
    """
    xMid = (parentPt[0] + cntrPt[0]) / 2.0                        # 计算中点的x坐标
    yMid = (parentPt[1] + cntrPt[1]) / 2.0                        # 计算中点的y坐标
    createPlot.ax1.text(xMid, yMid, txtString)                    # 在中点位置添加文本注释        

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """使用文本注解绘制树节点，利用 annotate 方法在指定位置绘制文本，并通过箭头连接父节点和子节点
    nodeTxt: 要显示在节点上的文本
    centerPt: 节点文本的中心位置，使用轴的分数坐标表示
    parentPt: 父节点的位置，用于确定箭头的方向，使用轴的分数坐标表示
    nodeType: 节点的样式，通常是一个字典，包含了边框的样式信息
    """
    createPlot.ax1.annotate(                             # 使用annotate方法在图表上添加注释
        nodeTxt,                                         # 节点文本
        xy=parentPt,                                     # 父节点位置
        xycoords='axes fraction',                        # 父节点位置的坐标系统；xycoords：这是一个参数，用于指定坐标值的参考坐标系；'axes fraction'：表示坐标值是相对于当前绘图区域（Axes）的比例值
        xytext=centerPt,                                 # 文本位置
        textcoords='axes fraction',                      # 文本位置的坐标系统
        va="center",                                     # 垂直对齐方式
        ha="center",                                     # 水平对齐方式
        bbox=nodeType,                                   # 节点样式
        arrowprops=arrow_args                            # 箭头属性
    )

def getNumLeafs(myTree):
    """计算树的叶节点数
    myTree: 输入的决策树
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]                      # 获取树的第一个键（特征名称）
    secondDict = myTree[firstStr]                          # 获取该特征对应的子树
    for key in secondDict.keys():                          # 遍历子树中的每个键（特征值）
        if isinstance(secondDict[key], dict):              # 如果该键对应的值是一个字典，则递归调用getNumLeafs函数
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1                                   # 否则，说明是叶节点，计数加1
    return numLeafs

def getTreeDepth(myTree):
    """计算树的深度
    myTree: 输入的决策树
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]                      # 获取树的第一个键（特征名称）
    secondDict = myTree[firstStr]                          # 获取该特征对应的子树
    for key in secondDict.keys():                          # 遍历子树中的每个键（特征值）
        if isinstance(secondDict[key], dict):              # 如果该键对应的值是一个字典，则递归调用getTreeDepth函数
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1                                   # 否则，说明是叶节点，深度为1
        if thisDepth > maxDepth:                            # 更新最大深度
            maxDepth = thisDepth
    return maxDepth

def plotTree(myTree, parentPt, nodeTxt):
    """绘制决策树
    myTree: 输入的决策树
    parentPt: 父节点的位置
    nodeTxt: 当前节点的文本
    """
    numLeafs = getNumLeafs(myTree)                         # 计算叶节点数
    depth = getTreeDepth(myTree)                           # 计算树的深度
    firstStr = list(myTree.keys())[0]                      # 获取树的第一个键（特征名称）
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)    # 计算当前节点的中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                 # 在父节点和子节点之间绘制文本
    plotNode(firstStr, cntrPt, parentPt, decisionNode)     # 绘制决策节点
    secondDict = myTree[firstStr]                          # 获取该特征对应的子树
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD    # 更新y坐标
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':           # test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        # recursion
        else:                                                # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW             # 更新x坐标
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode) # 绘制叶节点
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))   # 在父节点和子节点之间绘制文本
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD                     # 更新y坐标
    
def createPlot(inTree):
    """创建一个图表，并在其中绘制决策节点和叶节点
    inTree: 输入的决策树
    """
    fig = plt.figure(1, facecolor = 'white')            # 创建一个图表对象，背景颜色为白色
    fig.clf()                                           # 清除图表上的所有内容
    axprops = dict(xticks=[], yticks=[])                # 设置坐标轴属性，去掉刻度
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)                # 创建一个没有边框的子图;这种写法是全局变量，在其他函数中可直接调用
    plotTree.totalW = float(getNumLeafs(inTree))                             # 计算树的总宽度
    plotTree.totalD = float(getTreeDepth(inTree))                            # 计算树的总深度
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;               # 初始化x和y坐标
    plotTree(inTree, (0.5,1.0), '')                                          # 绘制决策树
    plt.show()

def retrieveTree(i):
    """测试函数：根据索引返回预定义的决策树
    i: 决策树的索引
    """
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, 
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: {'tail': {0: 'no', 1: 'yes'}}}}}}]
    return listOfTrees[i]

