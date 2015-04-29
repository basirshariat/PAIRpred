import numpy
import re
from codebase.data.dbd4.dbd4 import DBD4
import matplotlib.pyplot as plt

__author__ = 'basir'

seed = 1
number_of_samples = 20000
dbd4 = DBD4(size=number_of_samples, ratio=1, thresh=6, seed=seed)
lines = [line.strip() for line in open(dbd4.pairs_file)]
l_interacting_residues = []
r_interacting_residues = []
reversed_dictionary = {index: residue for residue, index in dbd4.residues.iteritems()}
ligand = set()
receptor = set()
i = 0
while i < len(lines) - 1:

    line = lines[i]
    (l_r, r_r, label) = re.split(' |_', line)
    if int(label) > 0:
        ligand.add(int(l_r))
        receptor.add(int(r_r))
        i += 1
    else:

        l_interacting_residues.append(len(ligand))
        r_interacting_residues.append(len(receptor))

        while int(label) < 0 and i < len(lines):
            (l_r, r_r, label) = re.split(' |_', lines[i])
            i += 1
        i -= 1
        ligand = set()
        receptor = set()

print l_interacting_residues
print r_interacting_residues
print numpy.mean(l_interacting_residues)
print numpy.mean(r_interacting_residues)
print numpy.std(l_interacting_residues)
print numpy.std(r_interacting_residues)

plt.hist(l_interacting_residues)
plt.title('Ligands')
plt.show()
plt.figure()
plt.hist(r_interacting_residues)
plt.title('Receptor')
plt.show()

