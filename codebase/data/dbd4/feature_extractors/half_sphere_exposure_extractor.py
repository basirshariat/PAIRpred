from datetime import datetime
import os
import numpy as np

from Bio.PDB.Polypeptide import standard_aa_names
from Bio.PDB import is_aa, Vector

from codebase.constants import half_sphere_exposure_directory
from codebase.data.dbd4.feature_extractors.residue_neighbourhood_extractor import ResidueNeighbourhoodExtractor
from codebase.pairpred.model.enums import Features
from codebase.pairpred.feature.abstract_feature_extractor import AbstractFeatureExtractor
from codebase.utils.printing import print_info_nn, print_info


__author__ = 'basir'


class HalfSphereExposureExtractor(AbstractFeatureExtractor):
    def __init__(self, database, **kwargs):
        super(HalfSphereExposureExtractor, self).__init__(database)
        self._residue_index_table = {}
        for index, amino_acid in enumerate(standard_aa_names):
            self._residue_index_table[amino_acid] = index
        ResidueNeighbourhoodExtractor(self._database, **kwargs).extract_feature()

    def extract_feature(self):
        counter = 0
        overall_time = datetime.now()
        number_of_amino_acids = len(standard_aa_names)
        print_info_nn(" >>> Adding Half Surface Exposure ... ".format(self._database.name))
        if not os.path.exists(self._get_dir_name()):
            os.makedirs(self._get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                hse_file = self._get_dir_name() + protein.name
                if not os.path.exists(hse_file + ".npy"):
                    counter += 1
                    if counter <= 15:
                        print_info_nn("{0}, ".format(protein.name))
                    else:
                        counter = 0
                        print_info("{0}".format(protein.name))
                    number_of_residues = len(protein.biopython_residues)
                    un = np.zeros(number_of_residues)
                    dn = np.zeros(number_of_residues)
                    uc = np.zeros((number_of_amino_acids, number_of_residues))
                    dc = np.zeros((number_of_amino_acids, number_of_residues))
                    for index, residue in enumerate(protein.biopython_residues):
                        u = self.get_side_chain_vector(residue)
                        if u is None:
                            un[index] = np.nan
                            dn[index] = np.nan
                            uc[:, index] = np.nan
                            dc[:, index] = np.nan
                        else:
                            residue_index = self._residue_index_table[residue.get_resname()]
                            uc[residue_index, index] += 1
                            dc[residue_index, index] += 1
                            neighbours_indices = protein.residues[index].get_feature(Features.RESIDUE_NEIGHBOURHOOD)
                            # print neighbours_indices
                            for neighbour_index in neighbours_indices:
                                if neighbour_index == -1:
                                    break
                                neighbour_residue = protein.biopython_residues[int(neighbour_index)]
                                if is_aa(neighbour_residue) and neighbour_residue.has_id('CA'):
                                    neighbour_vector = neighbour_residue['CA'].get_vector()
                                    residue_index = self._residue_index_table[neighbour_residue.get_resname()]
                                    if u[1].angle((neighbour_vector - u[0])) < np.pi / 2.0:
                                        un[index] += 1
                                        uc[residue_index, index] += 1
                                    else:
                                        dn[index] += 1
                                        dc[residue_index, index] += 1
                    uc = (uc / (1.0 + un)).T
                    dc = (dc / (1.0 + dn)).T
                    hse_array = np.hstack((uc, dc))
                    np.save(hse_file, hse_array)
                hse = np.load(hse_file + ".npy")
                for i in range(len(protein.residues)):
                    protein.residues[i].add_feature(Features.HALF_SPHERE_EXPOSURE, hse[i, :])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    @staticmethod
    def get_side_chain_vector(residue):
        """
        Find the average of the unit vectors to different atoms in the side chain
        from the c-alpha atom. For glycine the average of the N-Ca and C-Ca is
        used.
        Returns (C-alpha coordinate vector, side chain unit vector) for residue r
        """
        u = None
        gly = 0
        if is_aa(residue) and residue.has_id('CA'):
            ca = residue['CA'].get_coord()
            dv = np.array([ak.get_coord() for ak in residue.get_unpacked_list()[4:]])
            if len(dv) < 1:
                if residue.has_id('N') and residue.has_id('C'):
                    dv = [residue['C'].get_coord(), residue['N'].get_coord()]
                    dv = np.array(dv)
                    gly = 1
                else:
                    return None
            dv = dv - ca
            if gly:
                dv = -dv
            n = np.sum(abs(dv) ** 2, axis=-1) ** (1. / 2)
            v = dv / n[:, np.newaxis]
            u = (Vector(ca), Vector(v.mean(axis=0)))
        return u

    def _get_dir_name(self):
        return self._database.directory + half_sphere_exposure_directory
