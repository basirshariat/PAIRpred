from datetime import datetime
import os
import numpy as np

from Bio.PDB import DSSP

from codebase.constants import secondary_structure_directory, pdb_directory
from codebase.pairpred.model.enums import Features
from codebase.pairpred.feature.abstract_feature_extractor import AbstractFeatureExtractor
from codebase.utils.printing import print_info, print_info_nn


__author__ = 'basir'


class SecondaryStructureExtractor(AbstractFeatureExtractor):
    def extract_feature(self):
        print_info_nn(" >>> Adding secondary structure for database {0} ... ".format(self._database.name))
        overall_time = datetime.now()
        if not os.path.exists(self.__get_dir_name()):
            os.mkdir(self.__get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                dssp_file = self.__get_dir_name() + protein.name + ".npy"
                if not os.path.exists(dssp_file):
                    print_info_nn("... running DSSP for protein " + protein.name)
                    start_time = datetime.now()
                    dssp = DSSP(protein.structure[0], self._database.directory + pdb_directory + protein.name + ".pdb")
                    dssp_array = np.ndarray((len(protein.residues), 6))
                    for (i, res) in enumerate(protein.biopython_residues):
                        (_, _, cid, rid) = res.get_full_id()
                        key = (cid, rid)
                        if key in dssp:
                            dssp_array[i, 2:] = (dssp[key])[2:]
                        else:
                            dssp_array[i, 2:] = [0, 0, 0, 0]
                            # print_error("WTH")
                            # sys.exit(0)
                            # print('here')
                            # pdb.set_trace()
                            # self.SS[:, index] = np.nan
                            # self.ASA[index] = np.nan
                            # self.rASA[index] = np.nan
                            # self.Phi[index] = np.nan
                            # self.Psi[index] = np.nan
                    np.save(dssp_file, dssp_array)
                    print_info("took {0} seconds.".format((datetime.now() - start_time).seconds))
                dssp = np.load(dssp_file)
                for i, res in enumerate(protein.residues):
                    # (_, s, ASA, rASA, phi, psi)
                    res.add_feature(Features.ACCESSIBLE_SURFACE_AREA, dssp[i, 2])
                    res.add_feature(Features.RELATIVE_ACCESSIBLE_SURFACE_AREA, dssp[i, 3])
                    res.add_feature(Features.PHI, dssp[i, 4])
                    res.add_feature(Features.PSI, dssp[i, 5])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))

    def __get_dir_name(self):
        return self._database.directory + secondary_structure_directory