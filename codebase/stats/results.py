import cPickle
from PyML.evaluators.roc import plotROCs

__author__ = 'basir'
kernels = ["Profile", "Profile + Plain D2", "Profile + Plain D1", "Profile + Surface D2", "Profile + Surface D1",
               "Profile + Category"]

f = open('')
results = cPickle.load(f)
plotROCs(results,   descriptions=kernels, legendLoc=4)
f.close()
