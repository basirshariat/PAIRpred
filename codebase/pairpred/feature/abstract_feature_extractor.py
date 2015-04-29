from abc import ABCMeta, abstractmethod
import numpy as np

__author__ = 'basir'


class AbstractFeatureExtractor(object):
    __metaclass__ = ABCMeta
    _database = None

    def __init__(self, database):
        self._database = database

    @abstractmethod
    def extract_feature(self):
        pass

    @staticmethod
    def _normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm