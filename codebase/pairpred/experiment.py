from datetime import datetime
from numpy import ndarray
import cPickle
from numpy.random import mtrand
import math

from PyML import SVM
from PyML.evaluators.assess import cvFromFolds
from PyML.evaluators.roc import plotROCs
from enum import Enum

from codebase.constants import reports_directory
from codebase.data.dbd4.dbd4 import DBD4
from codebase.pairpred.model.enums import Features
from codebase.utils.printing import print_info, print_special


__author__ = 'basir'


class Classifier(Enum):
    SVM = 1
    NN = 2
    AE = 3


class Experiment:
    def __init__(self, features, database, classifier):
        self.features = features
        self.database = database
        self.classifier = classifier
        self.pyml_result = None
        self.run_parameters = None

    def run(self, **kwargs):
        self.run_parameters = kwargs
        data = self.database.get_pyml_dataset(self.features, True, **kwargs)
        data.attachKernel('gaussian', gamma=kwargs['gamma'], normalization='cosine')  # normalization='cosine'
        if self.classifier == Classifier.SVM:
            # svm = SVM()
            svm = SVM(optimizer='pegasos')
            training, testing = self.database.get_cv_folds(kwargs['folds'])
            self.pyml_result = cvFromFolds(svm, data, training, testing, numFolds=kwargs['folds'], verbose=False)
            self.get_rfpp()
        if kwargs['save']:
            self.__save_results()

    def __save_results(self):
        pass

    def compute_patches(self):
        if self.pyml_result is None:
            return None

    def get_rfpp(self):
        if self.pyml_result is None:
            return None
        example_range = self.database.complexes_example_range
        example_complex_map = {}
        for complex_name in example_range:
            interval = example_range[complex_name]
            for example in range(interval[0], interval[2]):
                example_complex_map[example] = complex_name

        example_index_map = self.database.get_pair_index_map()
        complex_length_folds = []
        for fold in self.pyml_result:
            complex_length_map = {}
            for i in range(len(fold.L)):
                complex_name = example_complex_map[example_index_map[fold.patternID[i]]]
                if complex_name not in complex_length_map:
                    complex_length_map[complex_name] = 0
                complex_length_map[complex_name] += 1
            complex_length_folds.append(complex_length_map)

        rfpp = {}
        fold_no = 0
        for fold in self.pyml_result:
            complex_performance_map = {}
            complex_length_map = complex_length_folds[fold_no]
            for i in range(len(fold.L)):
                pid = fold.patternID[i]
                complex_name = example_complex_map[example_index_map[pid]]
                if complex_name not in complex_performance_map:
                    example_no = complex_length_map[complex_name]
                    complex_performance_map[complex_name] = ndarray((example_no, 3))
                    complex_length_map[complex_name] = 0

                perf_table = complex_performance_map[complex_name]
                length = complex_length_map[complex_name]
                perf_table[length, :] = [int(fold.L[i]), 2 * int(fold.Y[i]) - 1, fold.decisionFunc[i]]
                complex_length_map[complex_name] += 1
                number_of_examples_in_complex = perf_table.shape[0]
                if complex_length_map[complex_name] == number_of_examples_in_complex:
                    sorted_perf = perf_table[(-perf_table[:, 2]).argsort()]
                    complex_performance_map[complex_name] = sorted_perf

            for complex_name in complex_performance_map:
                perf_table = complex_performance_map[complex_name]
                for i in range(perf_table.shape[0]):
                    if perf_table[i, 0] > 0 and perf_table[i, 1] > 0:
                        rfpp[complex_name] = (i+1, perf_table.shape[0])
                        break
            fold_no += 1
        average = 0
        for complex_name in rfpp:
            rank, n = rfpp[complex_name]
            percent = math.ceil((rank * 100) / n)
            average += percent
            print_info("{0} : {1}".format(complex_name, percent))

        print_info("Average RFPP {0}".format(average))
        return rfpp


def save_results(number_of_samples, results, feature_sets):
    kernels = [
        "Profile",
        "Profile + Plain D2",
        "Profile + Plain D1",
        "Profile + Surface D2",
        "Profile + Surface D1",
        "Profile + Category",
        # "Profile + SD +exp",
        "Profile + SD +exp"
    ]
    features_list_rep = []
    for feature_set in feature_sets:
        features_list_rep.append([])
        for feature in feature_set:
            features_list_rep[-1].append(feature.value)
    filename_prefix = reports_directory + "kernel/Complex-Wise-{0}-{1}".format(number_of_samples, features_list_rep)
    print filename_prefix
    print filename_prefix
    plotROCs(results, fileName=filename_prefix + ".pdf", descriptions=kernels, legendLoc=4)
    f = open(filename_prefix + ".cpickle", "wb")
    cPickle.dump(results, f)
    f.close()
    print "\\begin{table}[t]"
    print "\\centering"
    print "\\caption{Comparison of different kernels}"
    print "\\begin{tabular}{|c|c|c|c|c|c|}\\hline"
    print "Kernel & Sucess Rate & Balanced Success Rate & AUC & AUC 50 & Confusion Matrix\\\\ \\hline"
    for i in range(len(results)):
        r = results[i]
        matrix = r.getConfusionMatrix()
        print "{0} & {1:.4f} & {2:.4f} & {3:.4f} & {4:.4f} & " \
              "$\\left[ \\begin{9}{10} {5} & {6}  \\\\ {7} & {8} \\end{9}\\right]$  \\\\ \\hline" \
            .format(kernels[i],
                    r.getSuccessRate(),
                    r.getBalancedSuccessRate(),
                    r.roc,
                    r.roc50,
                    matrix[0][0],
                    matrix[0][1],
                    matrix[1][0],
                    matrix[1][1],
                    "{array}",
                    "{cc}")
    print "\\end{tabular}"
    print "\\end{table}"


def main():
    print_info("Starting the experiment")
    start_time = datetime.now()
    seed = 1
    number_of_samples = 500
    # number_of_samples = 20000
    dbd4 = DBD4(size=number_of_samples, ratio=1, thresh=6, seed=seed)
    mtrand.seed(seed)
    feature_sets = [
        [
            Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX,
            Features.WINDOWED_POSITION_SPECIFIC_FREQUENCY_MATRIX,
        ],
        # [
        #     Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX,
        #     Features.D2_PLAIN_SHAPE_DISTRIBUTION
        # ],
        # [
        #     Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX,
        #     Features.D1_PLAIN_SHAPE_DISTRIBUTION
        # ],
        # [
        #     Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX,
        #     Features.RELATIVE_ACCESSIBLE_SURFACE_AREA,
        #     Features.D2_SURFACE_SHAPE_DISTRIBUTION
        # ],
        # [
        #     Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX,
        #     Features.RELATIVE_ACCESSIBLE_SURFACE_AREA,
        #     Features.D1_SURFACE_SHAPE_DISTRIBUTION
        # ],
        # [
        #     Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX,
        #     Features.D2_CATEGORY_SHAPE_DISTRIBUTION
        # ],
        # [
        #     Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX,
        #     # Features.PROTRUSION_INDEX,
        #     # Features.B_VALUE,
        #     Features.HALF_SPHERE_EXPOSURE,
        #     Features.SECONDARY_STRUCTURE,
        #     Features.WINDOWED_POSITION_SPECIFIC_FREQUENCY_MATRIX,
        #     Features.POSITION_SPECIFIC_SCORING_MATRIX,
        #     Features.POSITION_SPECIFIC_FREQUENCY_MATRIX,
        #     Features.RELATIVE_ACCESSIBLE_SURFACE_AREA,
        #     # # Features.PHI,
        #     # # Features.PSI,
        #     # Features.RELATIVE_ACCESSIBLE_SURFACE_AREA,
        #     Features.D2_SURFACE_SHAPE_DISTRIBUTION,
        #     # Features.D1_SURFACE_SHAPE_DISTRIBUTION,
        #     # Features.D2_PLAIN_SHAPE_DISTRIBUTION,
        #     # Features.D1_SURFACE_SHAPE_DISTRIBUTION,
        #     Features.RESIDUE_DEPTH
        # ]
    ]
    results = []
    for feature_set in feature_sets:
        print_special("Feature set {0}".format(feature_set))
        e = Experiment(feature_set, dbd4, Classifier.SVM)
        e.run(number_of_bins=20, radius=15, number_of_samples=-1, seed=seed, gamma=0.5, save=True, folds=5, rASA=.5)
        results.append(e.pyml_result)
        print_info("Took {0} seconds.".format((datetime.now() - start_time).seconds))
    save_results(number_of_samples, results, feature_sets)


if __name__ == "__main__":
    main()
    # todo compute the relative size of neighbours to the protein size
