from datetime import datetime
from numpy.linalg import norm
import os
import numpy as np
from scipy.spatial.distance import cdist

from Bio.PDB import NeighborSearch
from Bio.PDB.ResidueDepth import min_dist
from numpy.random.mtrand import seed

from codebase.constants import d1_surface_directory, pdb_directory
from codebase.data.dbd4.feature_extractors.d1_base_shape_distribution import D1BaseShapeDistributionExtractor
from codebase.data.dbd4.feature_extractors.secondary_srtucture_extractor import SecondaryStructureExtractor
from codebase.pairpred.model.enums import Features
from codebase.tools_interface.msms import get_surface_atoms
from codebase.utils.printing import print_info, print_info_nn


__author__ = 'basir'


class D1SurfaceAtomsShapeDistributionExtractor(D1BaseShapeDistributionExtractor):
    def __init__(self, database, **kwargs):
        super(D1SurfaceAtomsShapeDistributionExtractor, self).__init__(database, **kwargs)
        self.rNH = kwargs['rNH']

    def extract_feature(self):
        seed(self.seed)
        print_info_nn(" >>> Adding D1 surface atoms shape distribution for {0} ... ".format(self._database.name))
        overall_time = datetime.now()
        if not os.path.exists(self._get_dir_name()):
            os.makedirs(self._get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                shape_dist_file = self._get_dir_name() + protein.name
                if not os.path.exists(shape_dist_file + ".npy"):
                    print_info("{0}".format(protein.name))
                    pdb_file_name = self._database.directory + pdb_directory + protein.name + '.pdb'
                    surface, normals = get_surface_atoms(pdb_file_name)
                    distributions = np.zeros((len(protein.residues), 2*(self.number_of_bins + 1)))

                    for i in range(len(protein.residues)):
                        residue = protein.residues[i]
                        distributions[i, :] = self.get_distributions(residue.center, surface, normals)
                    np.save(shape_dist_file, distributions)
                distributions = np.load(shape_dist_file + ".npy")
                for i in range(len(protein.residues)):
                    protein.residues[i].add_feature(Features.D1_SURFACE_SHAPE_DISTRIBUTION, distributions[i, :])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    def _get_directory(self):
        return d1_surface_directory

    def get_distributions(self, center, surface, normals):
        distances = cdist(surface, center)
        indices = np.where(distances > self.rNH)
        selected_normals, selected_coordinates = normals[indices], surface[indices]

        dist = norm(selected_coordinates-center, axis=1)
        dist = dist/np.max(dist)
        bins = np.linspace(0, 1, self.number_of_bins)
        indices = np.digitize(dist, bins)

        distribution = np.bincount(indices)
        distribution = self._normalize(distribution)

        return distances