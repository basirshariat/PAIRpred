from datetime import datetime
from numpy.linalg import norm
import os
import numpy as np

from Bio.PDB import ResidueDepth
from numpy.random.mtrand import choice

from codebase.constants import pdb_directory, residue_depth_directory
from codebase.data.dbd4.feature_extractors.d2_base_shape_distribution import D2BaseShapeDistributionExtractor
from codebase.pairpred.model.enums import Features
from codebase.utils.printing import print_info, print_info_nn, print_error


__author__ = 'basir'


class ResidueDepthExtractor(D2BaseShapeDistributionExtractor):
    def extract_feature(self):
        print_info_nn(" >>> Adding residue depth for database {0} ... ".format(self._database.name))
        overall_time = datetime.now()
        counter = 0
        if not os.path.exists(self._get_dir_name()):
            os.makedirs(self._get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                residue_depth_file = self._get_dir_name() + protein.name + ".npy"
                if not os.path.exists(residue_depth_file):
                    counter += 1
                    if counter <= 15:
                        print_info_nn("{0}, ".format(protein.name))
                    else:
                        counter = 0
                        print_info("{0}".format(protein.name))

                    pdb_file = self._database.directory + pdb_directory + protein.name + ".pdb"
                    rd = ResidueDepth(protein.structure[0], pdb_file)
                    rd_array = np.ndarray((len(protein.residues), 2))  # self.number_of_bins +
                    # surface = get_surface(pdb_file)
                    for (i, res) in enumerate(protein.biopython_residues):
                        (_, _, c, (h, rn, ic)) = res.get_full_id()
                        key = (c, (h, rn, ic))
                        if key in rd:
                            rdv = rd[key]
                            if rdv[0] is None:
                                rdv = (0, rdv[1])
                                print "WTH?"
                            if rdv[1] is None:
                                rdv = (rdv[0], 0)
                                print "WTH?"
                            rd_array[i, :2] = rdv
                        else:
                            print_error('WTH')
                            rd_array[i, :2] = [0, 0]
                            # rd_array[i, 2:] = self._compute_distribution_(surface, protein.residues[i].center)

                    np.save(residue_depth_file, rd_array)
                surface_features = np.load(residue_depth_file)
                for i, res in enumerate(protein.residues):
                    res.add_feature(Features.RESIDUE_DEPTH, self._normalize(surface_features[i, :2]))
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    def _compute_distribution_(self, coordinates, center):
        n = (coordinates.shape[0])
        if self.number_of_samples != -1:
            first = coordinates[np.ix_(choice(n, self.number_of_samples))]
            second = np.ones((self.number_of_samples, 3)) * center
        else:
            first = coordinates
            second = np.ones(coordinates.shape) * center
        dist = norm(first - second, axis=1)
        dist = dist / np.max(dist)
        bins = np.linspace(0, 1, self.number_of_bins)
        indices = np.digitize(dist, bins)
        distribution = np.bincount(indices)
        return self._normalize(distribution)[1:]

    def _get_directory(self):
        return residue_depth_directory