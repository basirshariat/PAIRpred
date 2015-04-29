from datetime import datetime
import os
import numpy as np

from codebase.constants import pdb_directory, stride_secondary_structure_directory, ss_abbreviations
from codebase.data.dbd4.tools.stride import stride_dict_from_pdb_file
from codebase.pairpred.model.enums import Features
from codebase.pairpred.feature.abstract_feature_extractor import AbstractFeatureExtractor
from codebase.utils.printing import print_info, print_info_nn


__author__ = 'basir'


class StrideSecondaryStructureExtractor(AbstractFeatureExtractor):
    def extract_feature(self):
        secondary_structure_dict = dict(zip(ss_abbreviations, range(len(ss_abbreviations))))
        print_info_nn(" >>> Adding secondary structure for database {0} ... ".format(self._database.name))
        overall_time = datetime.now()
        counter = 0
        if not os.path.exists(self.__get_dir_name()):
            os.mkdir(self.__get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                stride_x_file = self.__get_dir_name() + protein.name + ".npy"
                if not os.path.exists(stride_x_file):
                    counter += 1
                    if counter <= 15:
                        print_info_nn("{0}, ".format(protein.name))
                    else:
                        counter = 0
                        print_info("{0}".format(protein.name))

                    pdb_file = self._database.directory + pdb_directory + protein.name + ".pdb"
                    n = len(protein.residues)
                    stride_x = stride_dict_from_pdb_file(pdb_file)
                    stride_x_array = np.zeros((n, 11))
                    for index, residue in enumerate(protein.biopython_residues):
                        key = self.get_residue_id(residue.get_full_id())
                        if key in stride_x:
                            (_, s, phi, psi, asa, rasa) = stride_x[key]
                            if s not in secondary_structure_dict:
                                raise ValueError("unknown secondary structure! Add to dictionary!")
                            ss = np.zeros(len(secondary_structure_dict))
                            ss[secondary_structure_dict[s]] = 1
                            stride_x_array[index, :7] = ss
                            stride_x_array[index, 7] = phi
                            stride_x_array[index, 8] = psi
                            stride_x_array[index, 9] = asa
                            stride_x_array[index, 10] = rasa
                    np.save(stride_x_file, stride_x_array)
                stride_x = np.load(stride_x_file)
                for i, res in enumerate(protein.residues):
                    res.add_feature(Features.SECONDARY_STRUCTURE, stride_x[i, :7])
                    res.add_feature(Features.PHI, stride_x[i, 7])
                    res.add_feature(Features.PSI, stride_x[i, 8])
                    res.add_feature(Features.ACCESSIBLE_SURFACE_AREA, stride_x[i, 9])
                    res.add_feature(Features.RELATIVE_ACCESSIBLE_SURFACE_AREA, stride_x[i, 10])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    @staticmethod
    def get_residue_id(fid):
        """
        Given the full id of a residue, return the tuple id form
        """
        (_, _, cid, (_, residue_index, ri_num)) = fid
        return cid, str(residue_index) + ri_num.strip()  #

    def __get_dir_name(self):
        return self._database.directory + stride_secondary_structure_directory