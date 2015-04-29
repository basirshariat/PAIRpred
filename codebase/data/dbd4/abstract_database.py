
from abc import ABCMeta, abstractmethod

__author__ = 'basir'


class AbstractDatabase(object):
    __metaclass__ = ABCMeta
    name = ""

    @abstractmethod
    def get_pyml_dataset(self, features, paired_data=False, **kwargs):
        pass

    @abstractmethod
    def _save(self):
        pass

    @abstractmethod
    def _load(self, file_names):
        pass
