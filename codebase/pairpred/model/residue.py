import numpy as np
from codebase.utils.printing import print_info, print_warning, print_error

__author__ = 'basir'


class Residue:
    def __init__(self, residue):
        self.residue = residue
        self.computed_features = {}
        coordinates = [0, 0, 0]
        for atom in residue.get_list():
            coordinates += atom.get_coord()
        self.center = coordinates / len(residue.get_list())

    def get_computed_features(self):
        return self.computed_features.keys()

    def get_vector_form(self, features):
        uncomputed_features = set(features) - set(self.get_computed_features())
        if uncomputed_features != set([]):
            print_error(
                "Following features {0} is still not computed for this residue {1}".format(uncomputed_features, self))
            return None
        temp = None
        for feature in features:

            vector = self.computed_features[feature]
            if temp is None:
                temp = vector
            else:
                temp = np.hstack((vector, temp))
        return temp

    def add_feature(self, feature, vector_form):
        # if feature not in self.computed_features:
        self.computed_features[feature] = vector_form

    # else:
    # print_warning("Feature {0} for residue {1} already computed!".format(feature, self))

    def get_feature(self, feature):
        if feature not in self.computed_features:
            print_error("Feature {0} is not computed!".format(feature))
        else:
            return self.computed_features[feature]

    def get_coordinates(self):
        atoms = self.residue.get_list()
        coordinates = np.zeros((len(atoms), 3))
        for index, atom in enumerate(atoms):
            coordinates[index, :] = atom.get_coord()
        return coordinates