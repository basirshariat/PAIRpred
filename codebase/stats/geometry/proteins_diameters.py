import numpy
from codebase.data.dbd4.dbd4 import DBD4
import matplotlib.pyplot as plt

__author__ = 'basir'

seed = 1
number_of_samples = 20000
dbd4 = DBD4(size=number_of_samples, ratio=1, thresh=6, seed=seed)

l_diameters = []
r_diameters = []
for complex_name in dbd4.complexes:
    c = dbd4.complexes[complex_name]
    l_diameters.append(c.bound_formation.ligand.compute_diameter())
    r_diameters.append(c.bound_formation.receptor.compute_diameter())

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
