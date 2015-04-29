# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from codebase.constants import pdb_directory, dbd4_directory
# from codebase.tools_interface.msms import get_surface_atoms
from codebase.tools_interface.msms import get_surface_atoms


__author__ = 'basir'

pdb_file = dbd4_directory + pdb_directory + '1GHQ_l_b.pdb'
# protein = Protein(*read_pdb_file(pdb_file))
# rd = ResidueDepth(protein.structure[0], pdb_file)
#
surface = get_surface_atoms(pdb_file)

points = surface
xs = points[:, 0]
ys = points[:, 1]
zs = points[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()

print surface
