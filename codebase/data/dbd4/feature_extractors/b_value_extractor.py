from datetime import datetime
import os
import numpy as np

from codebase.constants import b_factor_directory
from codebase.pairpred.model.enums import Features
from codebase.pairpred.feature.abstract_feature_extractor import AbstractFeatureExtractor
from codebase.utils.printing import print_info_nn, print_info


__author__ = 'basir'


class BValueExtractor(AbstractFeatureExtractor):

    def extract_feature(self):
        counter = 0
        overall_time = datetime.now()
        print_info_nn(" >>> Adding B Factor ... ".format(self._database.name))
        if not os.path.exists(self._get_dir_name()):
            os.makedirs(self._get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                b_factor_filename = self._get_dir_name() + protein.name
                if not os.path.exists(b_factor_filename + ".npy"):
                    counter += 1
                    if counter <= 15:
                        print_info_nn("{0}, ".format(protein.name))
                    else:
                        counter = 0
                        print_info("{0}".format(protein.name))

                    b_factor_array = np.zeros(len(protein.residues))
                    for (index, residue) in enumerate(protein.biopython_residues):
                        b_factor_array[index] = max([atom.get_bfactor() for atom in residue])

                    np.save(b_factor_filename, b_factor_array)
                b_factor_array = np.load(b_factor_filename + ".npy")
                # print b_factor_array
                for i in range(len(protein.residues)):
                    protein.residues[i].add_feature(Features.B_VALUE, b_factor_array[i])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))



    def _get_dir_name(self):
        return self._database.directory + b_factor_directory
