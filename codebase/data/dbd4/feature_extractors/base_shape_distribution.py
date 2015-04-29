from abc import abstractmethod

from codebase.pairpred.feature.abstract_feature_extractor import AbstractFeatureExtractor


__author__ = 'basir'


class BaseShapeDistributionExtractor(AbstractFeatureExtractor):
    def __init__(self, database, **kwargs):
        super(BaseShapeDistributionExtractor, self).__init__(database)
        self.radius = kwargs['radius']
        self.number_of_bins = kwargs['number_of_bins']
        self.number_of_samples = kwargs['number_of_samples']
        self.seed = kwargs['seed']

    @abstractmethod
    def extract_feature(self):
        pass

    @abstractmethod
    def _get_directory(self):
        pass

    def _get_dir_name(self):
        return self._database.directory + self._get_directory() + "{0}-{1}-{2}-{3}/".format(
            self.radius, self.number_of_bins,  self.number_of_samples, self.seed)