import plotter
import trees

fr = open('Project1_Contact lens predictions\Database\lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 读取数据集

lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
lensesTree = trees.createTree(lenses, lensesLabels)  # 创建决策树

print(lensesTree)  # 打印决策树
print(plotter.createPlot(lensesTree))  # 可视化决策树

