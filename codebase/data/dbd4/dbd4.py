import copy
import glob
import os
from os.path import basename
from random import sample, seed
from scipy.spatial.distance import cdist
import warnings
from datetime import datetime
import cPickle
import numpy as np
import gc
from random import shuffle

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from PyML import VectorDataSet, PairDataSet
from codebase.constants import dbd4_directory, pickle_directory, \
    dbd4_sample_file_prefix, pdb_directory, dbd4_object_model_file
from codebase.data.dbd4.abstract_database import AbstractDatabase
from codebase.data.dbd4.dbd4_feature_extractor import DBD4FeatureExtractor
from codebase.pairpred.model.protein import Protein
from codebase.pairpred.model.protein_complex import ProteinComplex
from codebase.pairpred.model.protein_pair import ProteinPair
from codebase.utils.pdb import read_pdb_file
from codebase.utils.printing import print_info, print_info_nn


__author__ = 'basir'


class DBD4(AbstractDatabase):
    def __init__(self, directory=dbd4_directory, load_from=None, **args):
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
        self.examples = []
        self.example_complex = {}
        self.complexes = {}
        self.residues = {}
        self.complexes_example_range = {}
        self.name = "Docking Benchmark Dataset 4.0"
        self.interaction_threshold = 6
        self.samples = []
        if load_from is not None:
            self._load(load_from)
        else:
            self.positives_size = args['size']
            self.positive_to_negative_ratio = args['ratio']
            self.seed = args['seed']
            if os.path.exists(self.directory + pickle_directory + dbd4_object_model_file):
                self._load()
            else:
                self.__read_pdb_files()
                self.__extract_examples()
                self._save()

        self.__load_samples()

    def _save(self, file_name=None):
        """
        This function saves all the attributes of the class: positive and negative examples, ligands and receptors and
        complex names are saved in pickle format.

        """
        if not os.path.exists(self.directory + pickle_directory):
            os.mkdir(self.directory + pickle_directory)
        if file_name is None:
            object_model_file_name = self.directory + pickle_directory + dbd4_object_model_file
        else:
            object_model_file_name = file_name

        f = open(object_model_file_name, "wb")
        print_info_nn("Saving the object model into {0} ... ".format(object_model_file_name))
        start_time = datetime.now()
        cPickle.dump((self.directory,
                      self.complexes,
                      self.residues,
                      self.complexes_example_range,
                      self.examples,
                      self.example_complex), f)
        f.close()
        print_info("took {0} seconds.".format((datetime.now() - start_time).seconds))

    def _load(self, file_name=None):
        """
        This function load all the attributes of the class: positive and negative examples, ligands and receptors and
        complex names are saved in pickle format.

        """

        if file_name is None:
            object_model_file_name = self.directory + pickle_directory + dbd4_object_model_file
        else:
            object_model_file_name = file_name

        f = open(object_model_file_name)
        print_info_nn("Loading the object model from {0} ... ".format(object_model_file_name))
        start_time = datetime.now()
        (self.directory,
         self.complexes,
         self.residues,
         self.complexes_example_range,
         self.examples,
         self.example_complex) = cPickle.load(f)
        f.close()
        gc.collect()
        print_info("took {0} seconds.".format((datetime.now() - start_time).seconds))

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

    def __get_residue_index(self, residue):
        if residue not in self.residues:
            self.residues[residue] = len(self.residues)
        return self.residues[residue]

    def __extract_examples(self):
        """
        This function returns the set of all positive and negative examples from DBD4 dataset. In protein complex C,
        wth receptor R and ligand L, two residues r on R and r' on L are considered as a positive example if in the
        bound form they are nearer than the threshold distance. All other pairs (r,r') with r on R and r' on L are
        considered as negative examples. Extracted examples are saved in self.examples
        """
        print_info("Finding the positive and negative examples in DBD4 ... {0}".format(self.positives_size))
        start_time = datetime.now()
        counter = 1
        start_index = 0
        neg_no = 0
        pos_no = 0
        for complex_name in self.complexes.keys():
            print_info_nn("{0}/{1}... processing complex {2}".format(counter, len(self.complexes), complex_name))
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
                        unbound_ligand_res_index = self.__get_residue_index(unbound_ligand_res)
                        unbound_receptor_res_index = self.__get_residue_index(unbound_receptor_res)
                        if dist_mat.min() < self.interaction_threshold:
                            pos.append((unbound_ligand_res_index, unbound_receptor_res_index, +1))
                        else:
                            neg.append((unbound_ligand_res_index, unbound_receptor_res_index, -1))
            self.examples.extend(copy.copy(pos))
            self.examples.extend(copy.copy(neg))
            pos_no += len(pos)
            neg_no += len(neg)
            self.complexes_example_range[complex_name] = (
                start_index, start_index + len(pos), start_index + len(neg) + len(pos))
            print_info(" ( {0:03d}/{1:05d} ) -{2}".format(len(pos), len(neg), self.complexes_example_range[complex_name]))
            start_index += len(pos) + len(neg)
            counter += 1
            all_e = pos + neg
            for e in all_e:
                self.example_complex["{0}_{1}".format(e[0], e[1])] = complex_name

        print_info("Finding examples in DBD4 took " + str((datetime.now() - start_time).seconds) + " seconds. ")
        print_info("The total number of examples found: " + str(pos_no + neg_no))

    def get_pair_index_map(self):
        pair_index_map = {}
        lines = [line.strip() for line in open(self.sample_file)]
        counter = 0
        for line in lines:
            pid = line.split()[0]
            pair_index_map[pid] = counter
            counter += 1
        return pair_index_map

    def __get_file_name(self, prefix, extension):
        return self.directory + prefix + "{0}-{1}-{2}.{3}".format(self.positives_size, self.positive_to_negative_ratio,
                                                                  self.seed, extension)

    def __load_samples(self):
        """"
        Based on three provided parameters self.positives_size, self.positive_to_negative_ratio and self.seed it picks
        a number of samples from the pool of self.examples.
        """""
        seed(self.seed)
        self.pyml_pair_file = self.__get_file_name(dbd4_sample_file_prefix, "pair")
        self.sample_file = self.__get_file_name(pickle_directory, "cpickle")
        if os.path.exists(self.sample_file):
            with open(self.sample_file) as sample_file:
                self.samples = cPickle.load(sample_file)
        else:
            positive_indices = []
            negative_indices = []
            for i, example in enumerate(self.examples):
                if example[2] == 1:
                    positive_indices.append(i)
                else:
                    negative_indices.append(i)

            if self.positives_size != -1 and self.positives_size < len(positive_indices):
                p_indices = sample(positive_indices, self.positives_size)
            else:
                p_indices = positive_indices
            if -1 == self.positive_to_negative_ratio or int(self.positives_size * self.positive_to_negative_ratio) > len(
                    negative_indices):
                n_indices = negative_indices
            else:
                n_indices = sample(negative_indices, int(self.positives_size * self.positive_to_negative_ratio))
            # saving to files
            with open(self.pyml_pair_file, "wr") as pair_file:
                indices = copy.copy(p_indices)
                indices.extend(copy.copy(n_indices))
                for i in indices:
                    example = self.examples[i]
                    if example[2] > 0:
                        pair_file.write("{0}_{1} +1\n".format(*example))
                    else:
                        pair_file.write("{0}_{1} -1\n".format(*example))
            # interleaving positive and negative examples for when we call get_cv_folds
            # this is NECESSARY because pyml will complain if there is no example from a class
            # and this is very likely to happen because #positive << #negatives.
            # so dont remove this !
            p_head = 0
            n_head = 0
            for i in range(len(self.examples)):
                if i == p_indices[p_head] and p_head < len(p_indices)-1:
                    self.samples.append(p_indices[p_head])
                    p_head += 1
                if i == n_indices[n_head] and n_head < len(n_indices)-1:
                    self.samples.append(n_indices[n_head])
                    n_head += 1
            #self.samples = p_indices + n_indices

            with open(self.sample_file, "wr") as sample_file:
                cPickle.dump(self.samples, sample_file)

    def get_pyml_dataset(self, features, **kwargs):
        uncomputed_features = self.__get_uncomputed_features(features)
        if len(uncomputed_features) != 0:
            DBD4FeatureExtractor.extract(features, self, **kwargs)
        reversed_dictionary = {index: residue for residue, index in self.residues.iteritems()}
        x = np.zeros((len(self.residues), reversed_dictionary[1].get_vector_form(features).shape[0]))
        for i in range(len(self.residues)):
            vector_form = reversed_dictionary[i].get_vector_form(features)
            x[i, :] = vector_form
        data_set = VectorDataSet(x)
        return PairDataSet(self.pyml_pair_file, data=data_set)

    def __get_uncomputed_features(self, features):
        a_residue = self.residues.keys()[0]
        uncomputed_features = []
        computed_features = set(a_residue.get_computed_features())
        for f in features:
            if f not in computed_features:
                uncomputed_features.append(f)
        return uncomputed_features

    def get_cv_folds(self, number_of_folds):
        with open(self.sample_file) as file:
            self.samples = cPickle.load(file)
		
        shuffle(self.samples)
        complex_examples = {}
        average_no_examples = len(self.samples)/number_of_folds
        print "Average Number of Examples {0} ".format(average_no_examples)
        for example in self.samples:
            for complex_name in self.complexes_example_range:
                (pos_start, neg_start, end) = self.complexes_example_range[complex_name]
                if pos_start <= example < end:
                    if complex_name not in complex_examples:
                        complex_examples[complex_name] = []
                    complex_examples[complex_name].append(example)

        folds = []
        training = []
        testing = []
        for i in range(number_of_folds):
            folds.append([])
            testing.append([])
            training.append([])

        current_fold = 0
        current_length = 0
        index = 0
        l = 1
        labels = []
        for complex_name in complex_examples:
            if current_length + len(complex_examples[complex_name]) > average_no_examples:
                current_length = 0
                current_fold += 1
                labels.append(l)
                l = 1
                if current_fold == number_of_folds:
                    break
            for i in complex_examples[complex_name]:
                folds[current_fold].append(index)
                l *= self.examples[self.samples[index]][2]
                index += 1
                current_length += 1
         
        for fold in range(number_of_folds):
            testing[fold] = folds[fold]
            for i in range(number_of_folds):
                if i != fold:
                    training[fold].extend(folds[i])
        print "HERE COME THE LABELS: {0}".format(labels)
        return training, testing

if __name__ == "__main__":
    pos_examples_no = 10
    ratio_of_neg_to_pos_examples = 1
    pos_example_thresh = 6
    d = DBD4(size=pos_examples_no, ratio=ratio_of_neg_to_pos_examples, thresh=pos_example_thresh, seed=10)
