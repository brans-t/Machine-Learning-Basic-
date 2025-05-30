# -*- coding: utf-8 -*-
# digit recognition

from numpy import zeros
from os import listdir
from os.path import join
from kNN import classify0

def img2vector(filename):
    """32*32的二进制图像转换为1*1024的行向量"""
    returnVect = zeros((1, 1024))                # 创建一个1x1024的零向量
    try:
        fr = open(filename, 'r')                 # 打开文件,'r' 是文件打开模式，表示以只读模式打开文件
        for i in range(32):                      # 遍历32行
            lineStr = fr.readline()              # 读取每一行
            for j in range(32):                  # 遍历每一行的32个字符
                returnVect[0, 32 * i + j] = int(lineStr[j])   # 将字符转换为整数并存储到向量中
        fr.close()  # 关闭文件
    except Exception as e:
        print(f"Error reading file {filename}: {e}")          # 捕获并打印错误信息
    return returnVect                                         # 返回转换后的向量

def handwritingClassTest():
    """
    手写数字识别测试函数
    """
    hwLabels = []  # 创建一个列表来存储训练数据集的标签
    trainingFileList = listdir('Database\\trainingDigits')  # 获取训练数据集文件列表
    numTrainingSamples = len(trainingFileList)  # 获取训练数据集的样本数
    trainingMat = zeros((numTrainingSamples, 1024))  # 创建一个训练数据集矩阵：numTrainingSamples行，1024列

    # 遍历训练数据集文件
    for i in range(numTrainingSamples):
        fileNameStr = trainingFileList[i]  # 获取文件名
        fileStr = fileNameStr.split('.')[0]  # 去掉文件扩展名
        classNumStr = int(fileStr.split('_')[0])  # 提取类别标签
        hwLabels.append(classNumStr)  # 将类别标签添加到列表中
        trainingMat[i, :] = img2vector(join('Database\\trainingDigits', fileNameStr))  # 将图像转换为向量并存储到矩阵中;join 是 os.path.join 函数的简写，用于将多个路径组件组合成一个完整的路径

    testFileList = listdir('Database\\testDigits')  # 获取测试数据集文件列表
    errorCount = 0.0  # 初始化错误计数
    numTestSamples = len(testFileList)  # 获取测试数据集的样本数

    # 遍历测试数据集文件
    for i in range(numTestSamples):
        fileNameStr = testFileList[i]  # 获取文件名
        fileStr = fileNameStr.split('.')[0]  # 去掉文件扩展名
        classNumStr = int(fileStr.split('_')[0])  # 提取类别标签
        vectorUnderTest = img2vector(join('Database\\testDigits', fileNameStr))  # 将图像转换为向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  # 使用k-近邻分类器进行分类
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))  # 打印分类结果
        if (classifierResult != classNumStr):  # 如果分类结果不正确
            errorCount += 1  # 增加错误计数

    print("\nthe total number of errors is: %d" % errorCount)  # 打印总错误数
    print("\nthe total error rate is: %f" % (errorCount / float(numTestSamples)))  # 打印错误率

if __name__ == "__main__":
    handwritingClassTest()  # 调用手写数字识别测试函数