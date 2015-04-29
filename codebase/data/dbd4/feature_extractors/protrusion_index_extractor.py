from datetime import datetime
import os
import numpy as np

from codebase.constants import protrusion_index_directory, pdb_directory
from codebase.data.dbd4.tools.psaia import run_psaia
from codebase.pairpred.model.enums import Features
from codebase.pairpred.feature.abstract_feature_extractor import AbstractFeatureExtractor
from codebase.utils.printing import print_info_nn, print_info


__author__ = 'basir'


class ProtrusionIndexExtractor(AbstractFeatureExtractor):
    def __init__(self, database):
        super(ProtrusionIndexExtractor, self).__init__(database)

    @staticmethod
    def _normalize_features(casa, rasa, rrasa, rdpx, rcx, rhph):
        """
        Normalizes all PSAIA features to 0-1
        """
        rdpx_max = np.array([7.5131, 2.4013, 7.658, 1.8651, 8.0278, 7.0175])
        rdpx_min = -1.0
        rcx_max = np.array([11.153, 4.8229, 12.212, 4.4148, 19.84, 9.4199])
        rcx_min = -1.0
        rrasa_max = np.array([167.0, 368.04, 124.66, 316.96, 253.85])
        rrasa_min = 0.0
        casa_max = np.array([36661.0, 7584.9, 29756.0, 15550.0, 21489.0])
        casa_min = np.array([1412.0, 235.32, 1174.2, 362.0, 940.64])
        rasa_min = 0.0
        rasa_max = np.array([273.22, 134.51, 216.4, 173.26, 185.47])
        rhph_min = -4.5
        rhph_max = +4.5
        n_rdpx = (rdpx - rdpx_min) / ((rdpx_max - rdpx_min)[:, np.newaxis]).T

        n_rcx = (rcx - rcx_min) / ((rcx_max - rcx_min)[:, np.newaxis]).T
        n_rrasa = (rrasa - rrasa_min) / ((rrasa_max - rrasa_min)[:, np.newaxis]).T
        n_casa = ((casa - casa_min)[:, np.newaxis]).T / ((casa_max - casa_min)[:, np.newaxis]).T
        n_rasa = (rasa - rasa_min) / ((rasa_max - rasa_min)[:, np.newaxis]).T
        n_rhph = (rhph - rhph_min) / (rhph_max - rhph_min)
        protrusion_array = np.zeros(28)
        protrusion_array[:5] = n_casa
        protrusion_array[5:10] = n_rasa
        protrusion_array[10:15] = n_rrasa
        protrusion_array[15:21] = n_rdpx
        protrusion_array[21:27] = n_rcx
        protrusion_array[27] = n_rhph
        return protrusion_array

    def extract_feature(self):
        print_info_nn(" >>> Adding secondary structure for database {0} ... ".format(self._database.name))
        overall_time = datetime.now()
        counter = 0
        if not os.path.exists(self.__get_dir_name()):
            os.mkdir(self.__get_dir_name())
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                protrusion_file = self.__get_dir_name() + protein.name
                if not os.path.exists(protrusion_file+".npy"):
                    counter += 1
                    if counter <= 15:
                        print_info_nn("{0}, ".format(protein.name))
                    else:
                        counter = 0
                        print_info("{0}".format(protein.name))
                    pdb_file = self._database.directory + pdb_directory + protein.name + ".pdb"
                    result_dict = run_psaia(pdb_file)
                    protrusion_array = np.zeros((len(protein.residues), 5 + 5 + 5 + 6 + 6 + 1))
                    if result_dict is not None:
                        for index, residue in enumerate(protein.biopython_residues):
                            key = self.get_residue_id(residue.get_full_id())
                            if key in result_dict:
                                values = result_dict[key]
                                protrusion_array[index, :] = self._normalize_features(*values)
                            else:
                                print('key not found in PSAIA processing!')
                    np.save(protrusion_file, protrusion_array)
                protrusion_array = np.load(protrusion_file+".npy")
                for index, residue in enumerate(protein.residues):
                    residue.add_feature(Features.PROTRUSION_INDEX, protrusion_array[index, 21:])
        print_info("took {0} seconds.".format((datetime.now() - overall_time).seconds))


    @staticmethod
    def get_residue_id(full_id):
        """
        Given the full id of a residue, return the tuple id form
        """
        (_, _, cid, (_, residue_index, ri_num)) = full_id
        return cid, str(residue_index) + ri_num.strip()  #

    def __get_dir_name(self):
        return self._database.directory + protrusion_index_directory
