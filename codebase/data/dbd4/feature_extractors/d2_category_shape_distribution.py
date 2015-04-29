from datetime import datetime
import os
import numpy as np

from Bio.PDB import NeighborSearch
from numpy.random.mtrand import seed

from codebase.constants import d2_category_directory
from codebase.data.dbd4.feature_extractors.d2_base_shape_distribution import D2BaseShapeDistributionExtractor
from codebase.pairpred.model.enums import Features
from codebase.utils.printing import print_info, print_info_nn


__author__ = 'basir'

polar = 'polar'
charged = 'charged'
hydrophobic = 'hydrophobic'

categories = dict({
    'CYS': polar,
    'GLN': polar,
    'ASP': charged,
    'SER': polar,
    'VAL': hydrophobic,
    'LYS': charged,
    'PRO': hydrophobic,
    'THR': polar,
    'TRP': polar,
    'PHE': hydrophobic,
    'GLU': charged,
    'HIS': polar,
    'GLY': hydrophobic,
    'ILE': hydrophobic,
    'LEU': hydrophobic,
    'ARG': charged,
    'ALA': hydrophobic,
    'ASN': polar,
    'TYR': polar,
    'MET': polar,
})


class D2CategoryShapeDistributionExtractor(D2BaseShapeDistributionExtractor):
    def extract_feature(self):
        seed(self.seed)
        counter = 0
        print_info_nn(" >>> Adding D2 category based shape distribution for database {0} ... ".format(self._database.name))
        overall_time = datetime.now()
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
                    distributions = np.zeros((len(protein.residues), self.number_of_bins))
                    for i in range(len(protein.residues)):
                        residue = protein.residues[i]
                        nearby_residues = neighbour_search.search(residue.center, self.radius, "R")
                        distributions[i, :] = self._compute_distribution(nearby_residues)
                    np.save(shape_dist_file, distributions)
                distributions = np.load(shape_dist_file + ".npy")
                for i in range(len(protein.residues)):
                    protein.residues[i].add_feature(Features.D2_CATEGORY_SHAPE_DISTRIBUTION, distributions[i, :])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    def _compute_distribution_category(self, nearby_res):
        polar_charged = []
        hydrophobic_charged = []
        hydrophobic_polar = []
        all_cat = [polar_charged, hydrophobic_charged, hydrophobic_polar]
        for res in nearby_res:
            if categories[res.resname] == polar:
                polar_charged.append(res)
                hydrophobic_polar.append(res)
            if categories[res.resname] == hydrophobic:
                hydrophobic_charged.append(res)
                hydrophobic_polar.append(res)
            if categories[res.resname] == charged:
                polar_charged.append(res)
                hydrophobic_charged.append(res)
        final_distribution = []
        for category in all_cat:
            final_distribution.extend(list(self._compute_distribution(category)))
        return np.array(final_distribution)

    def _get_directory(self):
        return d2_category_directory
