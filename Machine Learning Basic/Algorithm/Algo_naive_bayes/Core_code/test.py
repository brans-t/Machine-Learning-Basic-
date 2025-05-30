import numpy as np

import Naive_Bayes
from Naive_Bayes import *
from numpy import *

# example

# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
#
# # print(listOPosts)
# # print(listClasses)
#
# print(myVocabList)
#
# trainMat = []
#
# for postinDoc in listOPosts:
#     trainMat.append(Naive_Bayes.setOfWords2Vec(myVocabList, postinDoc))
#
# p0V, p1V, pAb = Naive_Bayes.trainNB0(trainMat, listClasses)
#
# # print(p0V, p1V, pAb)
#
# # print(np.argmax(p1V))
# # print(max(p1V))
# # print(len(p1V))
# # print(sum(p1V))
# # a = int(np.argmax(p1V))
# # #
# # # print(a)
# # print(myVocabList[a])

Naive_Bayes.testingNB()