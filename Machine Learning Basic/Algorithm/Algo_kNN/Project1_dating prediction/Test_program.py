import date_pre as datepre
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import os

pathroot = os.path.dirname(os.path.abspath(__file__))  #获取当前文件路径
path = os.path.join(pathroot, 'Database\datingTestSet.txt')

datingDataMat, datingLabels = datepre.file2matrix(path)  #读取数据集
normMat, ranges, minVals = datepre.autoNorm(datingDataMat)  #归一化特征值

print(normMat)
# 绘制散点图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(normMat[:,1], normMat[:,2],
           s = 15.0*array(datingLabels), c = 15.0*array(datingLabels), cmap='viridis')
plt.show()