import numpy
import re
from scipy.spatial.distance import cdist
from scipy.spatial.qhull import ConvexHull
from codebase.data.dbd4.dbd4 import DBD4
import matplotlib.pyplot as plt

__author__ = 'basir'

seed = 1
number_of_samples = 20000
dbd4 = DBD4(size=number_of_samples, ratio=1, thresh=6, seed=seed)
lines = [line.strip() for line in open(dbd4.pairs_file)]
l_diameters = []
r_diameters = []
reversed_dictionary = {index: residue for residue, index in dbd4.residues.iteritems()}
ligand = []
receptor = []
i = 0
while i < len(lines) - 1:

    line = lines[i]
    (l_r, r_r, label) = re.split(' |_', line)
    if int(label) > 0:
        ligand.append(int(l_r))
        receptor.append(int(r_r))
        i += 1
    else:
        atoms = []
        number_of_atoms = 0
        for l_r in ligand:
            for atom in reversed_dictionary[l_r].residue.get_list():
                number_of_atoms += 1
        atoms = numpy.ndarray((number_of_atoms, 3))
        index = 0
        for l_r in ligand:
            for atom in reversed_dictionary[l_r].residue.get_list():
                atoms[index, :] = atom.get_coord()
                index += 1

        hull = ConvexHull(atoms)
        boundary_atoms = atoms[hull.vertices]
        diam = numpy.max(cdist(boundary_atoms, boundary_atoms))
        l_diameters.append(diam)
        # receptor
        number_of_atoms = 0
        for r_r in receptor:
            for atom in reversed_dictionary[r_r].residue.get_list():
                number_of_atoms += 1
        atoms = numpy.ndarray((number_of_atoms, 3))
        index = 0
        for r_r in receptor:
            for atom in reversed_dictionary[r_r].residue.get_list():
                atoms[index, :] = atom.get_coord()
                index += 1
        hull = ConvexHull(atoms)
        boundary_atoms = atoms[hull.vertices]
        diam = numpy.max(cdist(boundary_atoms, boundary_atoms))
        r_diameters.append(diam)

        while int(label) < 0 and i < len(lines):
            (l_r, r_r, label) = re.split(' |_', lines[i])
            i += 1
        i -= 1
        ligand = []
        receptor = []

print l_diameters
print r_diameters
print numpy.mean(l_diameters)
print numpy.mean(r_diameters)
print numpy.std(l_diameters)
print numpy.std(r_diameters)

plt.hist(l_diameters)
plt.title('Ligands')
plt.show()
plt.figure()
plt.hist(r_diameters)
plt.title('Receptor')
plt.show()

