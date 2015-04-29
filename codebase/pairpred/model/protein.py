from scipy.spatial.distance import cdist
from scipy.spatial.qhull import ConvexHull
from codebase.pairpred.model.residue import Residue
import numpy as np

__author__ = 'basir'


class Protein:

    def __init__(self, name, structure, residues, sequence, atoms):
        self.residues = []
        self.name = name
        for residue in residues:
            self.residues.append(Residue(residue))

        self.biopython_residues = residues
        self.sequence = sequence
        self.atoms = atoms
        self.structure = structure

    def compute_diameter(self):
        coordinates = np.ndarray((0, 3))
        for residue in self.residues:
            coordinates = np.vstack((coordinates, residue.get_coordinates()))
        hull = ConvexHull(coordinates)
        boundary = coordinates[hull.vertices]
        diam = np.max(cdist(boundary, boundary))
        return diam

    def get_atoms_coordinates(self):
        coordinates = np.ndarray((0, 3))
        for residue in self.residues:
            coordinates = np.vstack((coordinates, residue.get_coordinates()))
        return coordinates