from PyML import SparseDataSet, SVM

__author__ = 'basir'




data = SparseDataSet('data/heartSparse.data', labelsColumn=-1)
svm = SVM()
res = svm.cv(data, 5)
print res
# print data
# help(sequenceData.spectrum_data)