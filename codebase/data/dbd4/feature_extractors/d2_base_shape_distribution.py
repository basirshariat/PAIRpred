from abc import abstractmethod
from numpy.linalg import norm
import numpy as np
from scipy.spatial.distance import cdist

from numpy.random.mtrand import choice

from codebase.data.dbd4.feature_extractors.base_shape_distribution import BaseShapeDistributionExtractor


__author__ = 'basir'


class D2BaseShapeDistributionExtractor(BaseShapeDistributionExtractor):
    @abstractmethod
    def extract_feature(self):
        pass

    @abstractmethod
    def _get_directory(self):
        pass

    def _compute_distribution(self, nearby_res):
        atoms = []
        for r in nearby_res:
            atoms.extend(r.child_list)
        coordinates = np.zeros((len(atoms), 3))
        counter = 0
        for a in atoms:
            coordinates[counter] = a.get_coord()
            counter += 1
        n = len(atoms)
        if self.number_of_samples != -1:
            first = coordinates[np.ix_(choice(n, self.number_of_samples))]
            second = coordinates[np.ix_(choice(n, self.number_of_samples))]
            dist = norm(first - second, axis=1)
        else:
            first = coordinates
            second = coordinates
            dist_pair = cdist(first, second)
            dist = dist_pair.reshape((dist_pair.size,))

        dist = dist/np.max(dist)
        bins = np.linspace(0, 1, self.number_of_bins)
        indices = np.digitize(dist, bins)
        distribution = np.bincount(indices)
        distribution = self._normalize(distribution)
        return distribution[1:]