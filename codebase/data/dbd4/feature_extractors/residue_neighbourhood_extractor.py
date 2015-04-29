from datetime import datetime
import os
from scipy.spatial.distance import cdist
import numpy as np

from codebase.constants import residue_neighbourhood_directory, neighbourhood_threshold_key, \
    neighbourhood_sigma_key
from codebase.pairpred.model.enums import Features
from codebase.pairpred.feature.abstract_feature_extractor import AbstractFeatureExtractor
from codebase.utils.printing import print_info_nn, print_info


__author__ = 'basir'


class ResidueNeighbourhoodExtractor(AbstractFeatureExtractor):
    def __init__(self, database, **kwargs):
        super(ResidueNeighbourhoodExtractor, self).__init__(database)
        if neighbourhood_threshold_key in kwargs:
            self._threshold = kwargs[neighbourhood_threshold_key]
        else:
            self._threshold = 1e-3
        if neighbourhood_sigma_key in kwargs:
            self._sigma = kwargs[neighbourhood_sigma_key]
        else:
            self._sigma = 2.0

    def extract_feature(self):
        counter = 0
        print_info_nn(" >>> Adding Residue Neighbourhood ... ")
        overall_time = datetime.now()
        if not os.path.exists(self._get_dir_name()):
            os.makedirs(self._get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                residue_neighbourhood_file = self._get_dir_name() + protein.name
                if not os.path.exists(residue_neighbourhood_file + ".npy"):
                    counter += 1
                    if counter <= 15:
                        print_info_nn("{0}, ".format(protein.name))
                    else:
                        counter = 0
                        print_info("{0}".format(protein.name))
                    neighbourhood = []
                    max_length = 0
                    for i, query_residue in enumerate(protein.residues):
                        neighbourhood.append([])
                        for j, neighbour_residue in enumerate(protein.residues):
                            # if i == j:
                            #     continue
                            distance = cdist(query_residue.get_coordinates(), neighbour_residue.get_coordinates()).min()
                            similarity = np.exp(-(distance ** 2) / self._sigma)
                            if distance <= 7.5:
                                neighbourhood[-1].append(j)
                        if len(neighbourhood[-1]) > max_length:
                            max_length = len(neighbourhood[-1])
                    neighbourhood_array = -np.ones((len(protein.residues), max_length))
                    # print len(neighbourhood)
                    for i, residue_neighbourhood in enumerate(neighbourhood):
                        for j, neighbour_index in enumerate(neighbourhood[i]):
                            neighbourhood_array[i, j] = neighbourhood[i][j]
                        # print neighbourhood_array[i, :]
                    np.save(residue_neighbourhood_file, neighbourhood_array)
                neighbourhood_array = np.load(residue_neighbourhood_file+".npy")
                for index, residue in enumerate(protein.residues):
                    residue.add_feature(Features.RESIDUE_NEIGHBOURHOOD, neighbourhood_array[index, :])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    def _get_dir_name(self):
        return self._database.directory + residue_neighbourhood_directory + "{0}-{1}/".format(self._sigma,
                                                                                              self._threshold)