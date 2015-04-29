from datetime import datetime
import os
import numpy as np

from codebase.constants import unbound_sequence_directory, pssm_directory, psiblast_db_folder, psiblast_executable
from codebase.pairpred.model.enums import Features
from codebase.pairpred.feature.abstract_feature_extractor import AbstractFeatureExtractor
from codebase.utils.printing import print_info, print_info_nn, print_error


__author__ = 'basir'


class ProfileExtractor(AbstractFeatureExtractor):
    def __init__(self, database):
        super(ProfileExtractor, self).__init__(database)

    def extract_feature(self):
        self.__save_sequences_to_fasta()
        self.__compute_profiles()

    def __save_sequences_to_fasta(self):
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                sequence = protein.sequence
                fasta_file = self._database.directory + unbound_sequence_directory + protein.name + ".fasta"
                if os.path.exists(fasta_file):
                    continue
                print_info("... Saving sequence for protein " + protein.name)
                f = open(fasta_file, "w+")
                f.write(">{0}\n".format(protein.name))
                f.write(sequence + "\n")
                f.close()

    def __compute_profiles(self, db='nr', niter=3):
        print_info_nn(" >>> Adding the profile features for dataset {0} ...".format(self._database.name))
        start_time = datetime.now()
        for complex_name in self._database.complexes.keys():
            protein_complex = self._database.complexes[complex_name]
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                fasta_file = self._database.directory + unbound_sequence_directory + protein.name + ".fasta"
                output_file = self._database.directory + pssm_directory + protein.name
                if not os.path.exists(output_file + ".mat"):
                    print_info("... processing protein {0} ...    ".format(protein.name))
                    command = "cd {4} \n " \
                              "{5} " \
                              "-query {0} -db {1} -out {2}.psi.txt -num_iterations {3} -out_ascii_pssm {2}.mat" \
                        .format(fasta_file, db, output_file, niter, psiblast_db_folder, psiblast_executable)
                    print_info(command)
                    error_code = os.system(command)
                    if error_code == 0:
                        print_info('Successful!')
                    else:
                        print_error('Failed with error code {0}'.format(error_code))
                pssm, psfm, info = ProfileExtractor.__parse_pssm_file(output_file + ".mat")
                wpssm = ProfileExtractor.__get_wpsm(pssm)
                wpsfm = ProfileExtractor.__get_wpsm(psfm)
                for i, res in enumerate(protein.residues):
                    res.add_feature(Features.POSITION_SPECIFIC_SCORING_MATRIX, self._normalize(pssm[:, i]))
                    res.add_feature(Features.POSITION_SPECIFIC_FREQUENCY_MATRIX, self._normalize(psfm[:, i]))
                    res.add_feature(Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX, self._normalize(wpssm[:, i]))
                    res.add_feature(Features.WINDOWED_POSITION_SPECIFIC_FREQUENCY_MATRIX, self._normalize(wpsfm[:, i]))
        print_info("took {0} seconds.".format((datetime.now() - start_time).seconds))

    @staticmethod
    def __get_wpsm(psfm, window_size=5):
        window_size = int(window_size)
        (dimension, sequence_length) = psfm.shape
        pm = np.hstack((np.zeros((dimension, window_size)), psfm, np.zeros((dimension, window_size))))
        ws = 2 * window_size + 1
        wpssm = np.zeros((ws * dimension, sequence_length))
        for i in range(sequence_length):
            wpssm[:, i] = pm[:, i:i + ws].flatten('F')
        return wpssm

    @staticmethod
    def __parse_pssm_file(file_name):
        pssm = []
        psfm = []
        info = []
        try:
            for f in open(file_name, 'r'):
                f = f.split()
                if len(f) and f[0].isdigit():  # the first character must be a position
                    _pssm = [float(i) for i in f[2:22]]
                    pssm.append(_pssm)
                    # f=f[69:].split()
                    _psfm = [float(i) / 100.0 for i in f[22:42]]  # [float(i)/100.0 for i in f[:20]]#
                    psfm.append(_psfm)
                    # info.append(float(f[20]))
                    info.append(float(f[42]))
            z = (np.array(pssm).T, np.array(psfm).T, np.array(info))
        except IOError as e:
            print e
            z = None
        return z