from datetime import datetime
import os
import numpy as np

from Bio.PDB import NeighborSearch
from numpy.random.mtrand import seed

from codebase.constants import d2_directory
from codebase.data.dbd4.feature_extractors.d2_base_shape_distribution import D2BaseShapeDistributionExtractor
from codebase.pairpred.model.enums import Features
from codebase.utils.printing import print_info, print_info_nn


__author__ = 'basir'


class D2PlainShapeDistributionExtractor(D2BaseShapeDistributionExtractor):
    def extract_feature(self):
        seed(self.seed)
        counter = 0
        overall_time = datetime.now()
        print_info_nn(" >>> Adding D2 shape distribution for database {0} ... ".format(self._database.name))
        if not os.path.exists(self._get_dir_name()):
            os.makedirs(self._get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                shape_dist_file = self._get_dir_name() + protein.name
                if not os.path.exists(shape_dist_file+".npy"):
                    counter += 1
                    if counter <= 15:
                        print_info_nn("{0}, ".format(protein.name))
                    else:
                        counter = 0
                        print_info("{0}".format(protein.name))
                    atoms = protein.atoms
                    neighbour_search = NeighborSearch(atoms)
                    distributions = np.zeros((len(protein.residues), self.number_of_bins))
                    # distributions = np.zeros((len(protein.residues), self.number_of_bins+2))
                    for i in range(len(protein.residues)):
                        residue = protein.residues[i]
                        nearby_residues = neighbour_search.search(residue.center, self.radius, "R")
                        distributions[i, :] = self._compute_distribution(nearby_residues)
                        # distributions[i:, -1] = len(nearby_residues)
                    np.save(shape_dist_file, distributions)
                distributions = np.load(shape_dist_file+".npy")
                for i in range(len(protein.residues)):
                    protein.residues[i].add_feature(Features.D2_PLAIN_SHAPE_DISTRIBUTION, distributions[i, :])
                    # protein.residues[i].add_feature(Features.NUMBER_OF_NEIGHBOURS, distributions[i, -1])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    def _get_directory(self):
        return d2_directory