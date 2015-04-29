from numpy import ndarray, array
import numpy
from scipy.spatial.distance import cdist
from numpy.random.mtrand import rand
from scipy.spatial.qhull import ConvexHull
import matplotlib.pyplot as plt

__author__ = 'basir'

points = rand(10, 2)
print type(points)
print points.shape
hull = ConvexHull(points)
plt.plot(points[:, 0], points[:, 1], 'o')
# for simplex in hull.simplices:
# plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
# plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
boundary = array(points[hull.vertices, :])
print boundary
print numpy.max(cdist(boundary, boundary))
print numpy.max(cdist(points, points))
plt.show()
