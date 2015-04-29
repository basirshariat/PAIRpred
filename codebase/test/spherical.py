from scipy.spatial.distance import cdist
from numpy.random.mtrand import rand

___author__ = 'basir'
import warnings
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import numpy

octahedron_vertices = numpy.array([
    [1.0, 0.0, 0.0],  # 0
    [-1.0, 0.0, 0.0],  # 1
    [0.0, 1.0, 0.0],  # 2
    [0.0, -1.0, 0.0],  # 3
    [0.0, 0.0, 1.0],  # 4
    [0.0, 0.0, -1.0]  # 5
])
octahedron_triangles = numpy.array([
    [0, 4, 2],
    [2, 4, 1],
    [1, 4, 3],
    [3, 4, 0],
    [0, 2, 5],
    [2, 1, 5],
    [1, 3, 5],
    [3, 0, 5]])


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''

    lens = numpy.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def divide_all(vertices, triangles):
    # new_triangles = []
    new_triangle_count = len(triangles) * 4
    # Subdivide each triangle in the old approximation and normalize
    #  the new points thus generated to lie on the surface of the unit
    #  sphere.
    # Each input triangle with vertices labelled [0,1,2] as shown
    #  below will be turned into four new triangles:
    #
    #            Make new points
    #                 a = (0+2)/2
    #                 b = (0+1)/2
    #                 c = (1+2)/2
    #        1
    #       /\        Normalize a, b, c
    #      /  \
    #    b/____\ c    Construct new triangles
    #    /\    /\       t1 [0,b,a]
    #   /  \  /  \      t2 [b,1,c]
    #  /____\/____\     t3 [a,b,c]
    # 0      a     2    t4 [a,c,2]
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    a = ( v0 + v2 ) * 0.5
    b = ( v0 + v1 ) * 0.5
    c = ( v1 + v2 ) * 0.5
    normalize_v3(a)
    normalize_v3(b)
    normalize_v3(c)

    #Stack the triangles together.
    vertices = numpy.vstack((v0, b, a, b, v1, c, a, b, c, a, c, v2))
    #Now our vertices are duplicated, and thus our triangle structure are unnecesarry.
    return vertices, numpy.arange(len(vertices)).reshape((-1, 3))


def create_unit_sphere(recursion_level=2):
    vertex_array, index_array = octahedron_vertices, octahedron_triangles
    for i in range(recursion_level - 1):
        vertex_array, index_array = divide_all(vertex_array, index_array)
    return vertex_array, index_array


def vertex_array_only_unit_sphere(recursion_level=2):
    vertex_array, index_array = create_unit_sphere(recursion_level)
    if recursion_level > 1:
        return vertex_array.reshape((-1))
    else:
        return vertex_array[index_array].reshape((-1))

# http://en.wikipedia.org/wiki/Spherical_coordinate_system

# def main():
n = 1000
r = 1+0 * rand(n)
theta_0 = 0
theta = np.pi
# theta_0 = np.pi/20
# theta = 9*np.pi/10
phi_0 = 0
phi = np.pi

# thetas = np.linspace(0, np.pi, n)
thetas = theta * rand(n) + theta_0
# phis = list(np.linspace(0, 2 * np.pi, np.sqrt(n))) * int(np.sqrt(n))
phis = phi * rand(n) + phi_0
print np.mean(phis)
print np.std(phis)
plt.plot(phis)
plt.show()
# thetas = np.linspace(theta_0, theta, n)
# phis = np.linspace(phi_0, phi, n)

# for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
xs = r * np.cos(phis) * np.sin(thetas)
ys = r * np.sin(thetas) * np.sin(phis)
zs = r * np.cos(thetas)

points = np.ndarray((n, 3))
points[:, 0] = xs
points[:, 1] = ys
points[:, 2] = zs
# points, ii = create_unit_sphere(recursion_level=3)
n = points.shape[0]
print n
xs = points[:, 0]
ys = points[:, 1]
zs = points[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# plt.show()
plt.figure()
dist = cdist(points, points).reshape((n * n,))
number_of_bins = 10
bins = np.linspace(0, 1, number_of_bins)
indices = np.digitize(dist, bins)
distribution = np.bincount(indices)[1:-1]
print distribution
norm = np.linalg.norm(distribution)
distribution = list(distribution) / norm
# distribution = distribution
plt.plot(distribution)
plt.show()
