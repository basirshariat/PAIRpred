# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:26:38 2012

@author: fayyaz
"""
import numpy as np
from scipy import spatial
import itertools
import os
from PyML import *


def getPairData(pts=100):
    """
    Returns a PyML PairDataSet object.
    Generates pts random points and if two points are within a distance of
    1.0, then that pair is defined to be in the positive class, else neg.
    """
    fname = 'tempair.tmp'
    X = 2 * np.random.random((pts, 2)) - 1
    D = spatial.distance.cdist(X, X)
    (pr, pc) = np.where(D < 1.0)
    p = zip(pr, pc)
    n = set(itertools.product(range(X.shape[0]), range(X.shape[0]))).difference(p)
    f = open(fname, 'w')
    try:
        for e in p:
            s = str(e[0] + 1) + '_' + str(e[1] + 1) + ' +1\n'
            f.write(s)
        for e in n:
            s = str(e[0] + 1) + '_' + str(e[1] + 1) + ' -1\n'
            f.write(s)
    finally:
        f.close()
    pD = PairDataSet(fname, data=SparseDataSet(X))
    # os.remove(fname)
    return pD


print '#########################TRAINING####################'
tr_pD = getPairData()
s = SVM(ker.Gaussian(gamma=1.0))
# s.train(tr_pD)
# z = s.test(tr_pD)
# print z
# print '#########################TESTING####################'
# tt_pD = getPairData(10)
# z = s.test(tt_pD)
# print z
print s.cv(tr_pD, 5)
