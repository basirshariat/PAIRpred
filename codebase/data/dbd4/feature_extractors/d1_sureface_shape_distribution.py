from datetime import datetime
import os
import numpy as np

from Bio.PDB import NeighborSearch
from numpy.random.mtrand import seed

from codebase.constants import d1_surface_directory
from codebase.data.dbd4.feature_extractors.d1_base_shape_distribution import D1BaseShapeDistributionExtractor
from codebase.data.dbd4.feature_extractors.secondary_srtucture_extractor import SecondaryStructureExtractor
from codebase.pairpred.model.enums import Features
from codebase.utils.printing import print_info, print_info_nn


__author__ = 'basir'


class D1SurfaceShapeDistributionExtractor(D1BaseShapeDistributionExtractor):
    def __init__(self, database, **kwargs):
        super(D1SurfaceShapeDistributionExtractor, self).__init__(database, **kwargs)
        self.rASA_threshold = kwargs['rASA']
        SecondaryStructureExtractor(self._database).extract_feature()

    def extract_feature(self):
        seed(self.seed)
        print_info_nn(" >>> Adding D1 surface shape distribution for database {0} ... ".format(self._database.name))
        overall_time = datetime.now()
        counter = 0
        if not os.path.exists(self._get_dir_name()):
            os.makedirs(self._get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                shape_dist_file = self._get_dir_name() + protein.name
                if not os.path.exists(shape_dist_file + ".npy"):
                    counter += 1
                    if counter <= 15:
                        print_info_nn("{0}, ".format(protein.name))
                    else:
                        counter = 0
                        print_info("{0}".format(protein.name))
                    atoms = protein.atoms
                    neighbour_search = NeighborSearch(atoms)
                    distributions = np.zeros((len(protein.residues), self.number_of_bins + 1))
                    for i in range(len(protein.residues)):
                        residue = protein.residues[i]
                        nearby_residues = [protein.biopython_residues[i]]
                        temp_nearby_residues = neighbour_search.search(residue.center, self.radius, "R")
                        for nearby_residue in temp_nearby_residues:
                            if nearby_residue not in protein.biopython_residues:
                                continue
                            residues_index = protein.biopython_residues.index(nearby_residue)
                            residue = protein.residues[residues_index]

                            if residue.get_feature(Features.RELATIVE_ACCESSIBLE_SURFACE_AREA) >= self.rASA_threshold:
                                nearby_residues.append(nearby_residue)
                        distributions[i, :] = self._compute_distribution(nearby_residues, residue.center)
                    np.save(shape_dist_file, distributions)
                distributions = np.load(shape_dist_file + ".npy")
                for i in range(len(protein.residues)):
                    protein.residues[i].add_feature(Features.D1_SURFACE_SHAPE_DISTRIBUTION, distributions[i, :])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    def _get_directory(self):
        return d1_surface_directory