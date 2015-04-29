import copy
import glob
import os
from os.path import basename
from random import sample
from scipy.spatial.distance import cdist
import warnings
from math import floor
from datetime import datetime
import cPickle
import numpy as np
import gc
import subprocess

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from PyML import VectorDataSet, PairDataSet

from codebase.constants import dbd4_directory, pickle_directory, dbd4_object_model_file_prefix, \
    dbd4_pair_file_prefix, pdb_directory
from codebase.data.dbd4.abstract_database import AbstractDatabase
from codebase.data.dbd4.dbd4_feature_extractor import DBD4FeatureExtractor
from codebase.data.dbd4.example import ResidueInteraction
from codebase.pairpred.model.protein import Protein
from codebase.pairpred.model.protein_complex import ProteinComplex
from codebase.pairpred.model.protein_pair import ProteinPair
from codebase.utils.pdb import read_pdb_file
from codebase.utils.printing import print_info, print_info_nn


__author__ = 'basir'


class DBD4(AbstractDatabase):
    def __init__(self, directory=dbd4_directory, load_from=None, generate_examples=False, **args):
        """
            Either loads or creates the data model and training/testing examples for DBD4 Benchmark.

        @param directory(str): string Working directory that pdb files for DBD 4.0 reside on.
        @param load_from(str): address of the pickle file to load the data model from. If this argument is not specified
        @param generate_examples(bool): regenerates the examples even if the files already exists.
        @kwarg size(int): Minimum number of positive examples to be extracted from the dataset.
         (This is obviously bound from top by the maximum number of examples in the dataset.)
        @kwarg ratio(float): Ratio of number of positive examples to the number of negative examples.
        @kwarg thresh(float): Distance in angstrom for which a pair of residues is considered as interacting pair
         (a positive example). any pair with more distance than this number is considered as a negative example.
        @kwarg seed(hashable): Random seed used to sample negative examples.
        """
        self.directory = directory
        self.positives = {}
        self.negatives = {}
        self.complexes = {}
        self.residues = {}
        self.sampled_negative = {}
        self.complexes_example_range = {}
        self.name = "Docking Benchmark Dataset 4.0"
        self.pairs_file = ""

        if load_from is not None:
            self._load(load_from)
        else:
            self.positives_size = args['size']
            self.positive_to_negative_ratio = args['ratio']
            self.interaction_threshold = args['thresh']
            self.seed = args['seed']
            if self.__files_already_exist() and not generate_examples:
                self._load()
            else:
                self.__read_pdb_files()
                self.__extract_examples()
                self._save()

    def __read_pdb_files(self):
        print_info("Parsing the pdb files in directory {0} ....".format(os.path.abspath(self.directory)))
        ligand_bound_files = glob.glob(self.directory + pdb_directory + "*_l_b.pdb")
        ligand_bound_files.sort()
        counter = 0
        for ligand_bound_file in ligand_bound_files:
            complex_name = basename(ligand_bound_file).replace("_l_b.pdb", "")
            receptor_bound_file = ligand_bound_file.replace("_l_b.pdb", "_r_b.pdb")
            ligand_unbound_file = ligand_bound_file.replace("_l_b.pdb", "_l_u.pdb")
            receptor_unbound_file = ligand_bound_file.replace("_l_b.pdb", "_r_u.pdb")

            print_info("Reading complex " + complex_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PDBConstructionWarning)
                ligand_bound = Protein(*read_pdb_file(ligand_bound_file))
                receptor_bound = Protein(*read_pdb_file(receptor_bound_file))
                ligand_unbound = Protein(*read_pdb_file(ligand_unbound_file))
                receptor_unbound = Protein(*read_pdb_file(receptor_unbound_file))
                bound_formation = ProteinPair(ligand_bound, receptor_bound)
                unbound_formation = ProteinPair(ligand_unbound, receptor_unbound)
                self.complexes[complex_name] = ProteinComplex(complex_name, unbound_formation, bound_formation)

            counter += 1
        print_info("Total number of complexes processed : " + str(counter))

    def __extract_examples(self):
        """
        This function returns a set of positive and negative examples from DBD4 dataset. In protein complex C,
        wth receptor R and ligand L, two residues r on R and r' on L are considered as a positive example if in the
        bound form they are nearer than the threshold distance. All other pairs (r,r') with r on R and r' on L are
        considered as negative examples. Extracted examples are saved in two dictionaries positives and negatives.
        """
        unique_residues = set()
        print_info("Finding the positive and negative examples in DBD4 ... {0}".format(self.positives_size))
        start_time = datetime.now()
        pos_no = 0
        neg_no = 0
        counter = 1
        for complex_name in self.complexes.keys():
            print_info("{1}/{2}... processing complex {3}".format(counter, len(self.complexes), complex_name))
            protein_complex = self.complexes[complex_name]
            bound_ligand_bio_residues = protein_complex.bound_formation.ligand.biopython_residues
            bound_receptor_bio_residues = protein_complex.bound_formation.receptor.biopython_residues
            bound_ligand_residues = protein_complex.bound_formation.ligand.residues
            bound_receptor_residues = protein_complex.bound_formation.receptor.residues
            pos = []
            neg = []
            for i in range(len(bound_ligand_bio_residues)):
                for j in range(len(bound_receptor_bio_residues)):
                    bound_ligand_residue = bound_ligand_bio_residues[i]
                    bound_receptor_residue = bound_receptor_bio_residues[j]
                    l_atoms = [atom.get_coord() for atom in bound_ligand_residue.get_list()]
                    r_atoms = [atom.get_coord() for atom in bound_receptor_residue.get_list()]
                    dist_mat = cdist(l_atoms, r_atoms)
                    ligand_b2u = protein_complex.ligand_bound_to_unbound
                    receptor_b2u = protein_complex.receptor_bound_to_unbound
                    # if the residues have an unbound counterpart
                    # this is due to the fact that the unbound and bound formations may have slightly different residues
                    if bound_ligand_residues[i] in ligand_b2u and bound_receptor_residues[j] in receptor_b2u:
                        unbound_ligand_res = ligand_b2u[bound_ligand_residues[i]]
                        unbound_receptor_res = receptor_b2u[bound_receptor_residues[j]]
                        unique_residues.add(unbound_ligand_res)
                        unique_residues.add(unbound_receptor_res)
                        if dist_mat.min() < self.interaction_threshold:
                            pos.append(ResidueInteraction(complex_name, unbound_ligand_res, unbound_receptor_res))
                        else:
                            neg.append(ResidueInteraction(complex_name, unbound_ligand_res, unbound_receptor_res))
            if pos_no + len(pos) > self.positives_size:
                pos = pos[:(self.positives_size - pos_no)]

            self.positives[complex_name] = copy.copy(pos)
            random_indices = sample(range(len(neg)), int(floor(len(pos) * self.positive_to_negative_ratio)))
            self.sampled_negative[complex_name] = random_indices
            self.negatives[complex_name] = [neg[x] for x in random_indices]
            counter += 1
            pos_no += len(self.positives[complex_name])
            neg_no += len(self.negatives[complex_name])
            if pos_no >= self.positives_size:
                break
        self.__add_residues(list(unique_residues))
        print_info("Finding examples in DBD4 took " + str((datetime.now() - start_time).seconds) + " seconds. ")
        print_info("The total number of examples found: " + str(pos_no + neg_no))

    def _save(self):
        """
        This function saves all the attributes of the class: positive and negative examples, ligands and receptors and
        complex names are saved in pickle format.

        """
        object_model_file_name = self.__get_file_name(pickle_directory + dbd4_object_model_file_prefix, "cpickle")
        f = open(object_model_file_name, "wb")
        print_info_nn("Saving the object model into {0} ... ".format(object_model_file_name))
        start_time = datetime.now()
        self.pairs_file = self.create_pairs_files()
        cPickle.dump((self.directory,
                      self.complexes,
                      self.residues,
                      self.pairs_file,
                      self.complexes_example_range,
                      self.sampled_negative), f)
        f.close()
        print_info("took {0} seconds.".format((datetime.now() - start_time).seconds))

    def get_pair_index_map(self):
        pair_index_map = {}
        lines = [line.strip() for line in open(self.pairs_file)]
        counter = 0
        for line in lines:
            pid = line.split()[0]
            pair_index_map[pid] = counter
            counter += 1
        return pair_index_map

    def _load(self, file_names=None):
        """
        This function load all the attributes of the class: positive and negative examples, ligands and receptors and
        complex names are saved in pickle format.

        """

        if file_names is None:
            object_model_file_name = self.__get_file_name(pickle_directory + dbd4_object_model_file_prefix, "cpickle")
        else:
            object_model_file_name = file_names[0]

        f = open(object_model_file_name)
        print_info_nn("Loading the object model from {0} ... ".format(object_model_file_name))
        start_time = datetime.now()
        (self.directory,
         self.complexes,
         self.residues,
         self.pairs_file,
         self.complexes_example_range,
         self.sampled_negative) = cPickle.load(f)
        f.close()
        gc.collect()
        print_info("took {0} seconds.".format((datetime.now() - start_time).seconds))

    def __get_file_name(self, prefix, extension):
        return self.directory + prefix + "{0}-{1}-{2}-{3}.{4}".format(self.positives_size,
                                                                      self.positive_to_negative_ratio,
                                                                      self.interaction_threshold,
                                                                      self.seed,
                                                                      extension)

    def create_pairs_files(self):
        pair_file_name = self.__get_file_name(dbd4_pair_file_prefix, "pair")
        if os.path.exists(pair_file_name):
            return pair_file_name
        pair_file = open(pair_file_name, "wr")
        start_index = 0L
        for complex_name in self.complexes.keys():
            pos_index = 0L
            neg_index = 0L
            if complex_name not in self.positives:
                continue

            for positives_example in self.positives[complex_name]:
                ligand_index = str(self.residues[positives_example.ligand_residue])
                receptor_index = str(self.residues[positives_example.receptor_residue])
                pair_file.write("{0}_{1} +1\n".format(ligand_index, receptor_index))
                pos_index += 1

            for negatives_example in self.negatives[complex_name]:
                ligand_index = str(self.residues[negatives_example.ligand_residue])
                receptor_index = str(self.residues[negatives_example.receptor_residue])
                pair_file.write("{0}_{1} -1\n".format(ligand_index, receptor_index))
                neg_index += 1
            self.complexes_example_range[complex_name] = (start_index,
                                                          start_index + pos_index,
                                                          start_index + pos_index + neg_index)
            start_index += neg_index + pos_index
        pair_file.close()
        return pair_file_name

    def get_pyml_dataset(self, features, paired_data=False, **kwargs):
        uncomputed_features = self.__get_uncomputed_features(features)
        if len(uncomputed_features) != 0:
            DBD4FeatureExtractor.extract(features, self, **kwargs)
        reversed_dictionary = {index: residue for residue, index in self.residues.iteritems()}
        x = np.zeros((len(self.residues), reversed_dictionary[1].get_vector_form(features).shape[0]))
        for i in range(len(self.residues)):
            vector_form = reversed_dictionary[i + 1].get_vector_form(features)
            x[i, :] = vector_form
        data_set = VectorDataSet(x)
        if paired_data:
            pair_file_name = self.create_pairs_files()
            return PairDataSet(pair_file_name, data=data_set)
        else:
            return data_set

    def __add_residues(self, residue_list):
        for residue in residue_list:
            self.residues[residue] = len(self.residues) + 1

    def __files_already_exist(self):
        object_model_file_name = self.__get_file_name(pickle_directory + dbd4_object_model_file_prefix, "cpickle")
        examples_file_name = self.__get_file_name(dbd4_pair_file_prefix, "pair")
        return os.path.exists(object_model_file_name) and os.path.exists(examples_file_name)

    def __get_uncomputed_features(self, features):
        # a_complex = self.complexes[self.complexes.keys()[-1]]
        # a_residue = a_complex.unbound_formation.receptor.residues[0]
        a_residue = self.residues.keys()[0]
        uncomputed_features = []
        computed_features = set(a_residue.get_computed_features())
        for f in features:
            if f not in computed_features:
                uncomputed_features.append(f)
        return uncomputed_features

    def get_cv_folds(self, number_of_folds):
        if len(self.residues) == 0 or len(self.complexes_example_range) == 0:
            raise StandardError("examples information does not exist!")
        training = []
        testing = []
        number_of_examples = int(subprocess.check_output('wc -l %s' % self.pairs_file, shell=True).strip().split()[0])
        # print ">>>>" + str(number_of_examples)
        average_size_of_folds = 0
        for complex_name in self.sampled_negative.keys():
            interval = self.complexes_example_range[complex_name]
            average_size_of_folds += interval[1] - interval[0] + len(self.sampled_negative[complex_name])

        average_size_of_folds /= number_of_folds
        # average_size_of_folds = number_of_examples / number_of_folds
        # print "Average number of examples {0}".format(average_size_of_folds)
        folds = []
        for i in range(number_of_folds):
            folds.append([])
            testing.append([])
            training.append([])
        current_fold = 0
        current_length = 0
        for complex_name in self.sampled_negative.keys():
            interval = self.complexes_example_range[complex_name]
            interval_length = interval[1] - interval[0] + len(self.sampled_negative[complex_name])
            if current_length + interval_length > average_size_of_folds:
                current_length = 0
                current_fold += 1
                if current_fold == number_of_folds:
                    break
            folds[current_fold].append(complex_name)
            current_length += interval_length

        for fold in range(number_of_folds):
            testing_complexes = folds[fold]
            training_complexes = []
            for i in range(number_of_folds):
                if i != fold:
                    training_complexes.extend(folds[i])
            for complex_name in testing_complexes:
                pos_start, pos_end, neg_end = self.complexes_example_range[complex_name]
                for example_id in range(pos_start, neg_end):
                    testing[fold].append(example_id)
            for complex_name in training_complexes:
                pos_start, pos_end, neg_end = self.complexes_example_range[complex_name]
                for example_id in range(pos_start, pos_end):
                    training[fold].append(example_id)
                # training[fold].extend(self.sampled_negative[complex_name])
                training[fold] = list(set(range(number_of_examples)) - set(testing[fold]))

        return training, testing


if __name__ == "__main__":
    pos_examples_no = 500
    ratio_of_neg_to_pos_examples = 1
    pos_example_thresh = 6
    d = DBD4(size=pos_examples_no, ratio=ratio_of_neg_to_pos_examples, thresh=pos_example_thresh, seed=10)
