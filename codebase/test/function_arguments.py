from datetime import datetime

__author__ = 'basir'


def foo(*args, **kargs):
    # print a
    print args
    print kargs

foo('a', datetime.now(), f=1,f1=2,g='3')
