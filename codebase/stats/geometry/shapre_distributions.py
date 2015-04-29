from __future__ import division
from scipy.spatial.distance import cdist
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from numpy.random.mtrand import randint
from Bio.PDB import NeighborSearch
import matplotlib.pyplot as plt

from codebase.constants import reports_directory
from codebase.data.dbd4.feature_extractors.secondary_srtucture_extractor import SecondaryStructureExtractor
from codebase.pairpred.model.enums import Features
from codebase.data.dbd4.dbd4 import DBD4
from codebase.utils.printing import print_info

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D


__author__ = 'basir'


class GetOutOfLoop(Exception):
    pass


def get_coords(neighbour_search, protein, residue, threshold, surface):
    coordinates = []
    tmp_nearby_res = neighbour_search.search(residue.center, 15, "R")
    nearby_residues = []
    if surface:
        for nearby_residue in tmp_nearby_res:
            if nearby_residue not in protein.biopython_residues:
                continue
            residues_index = protein.biopython_residues.index(nearby_residue)
            residue = protein.residues[residues_index]
            if residue.get_feature(Features.RELATIVE_ACCESSIBLE_SURFACE_AREA) >= threshold:
                nearby_residues.append(nearby_residue)
    else:
        nearby_residues = tmp_nearby_res

    for r in nearby_residues:
        coordinates.extend([atom.get_coord() for atom in r.get_list()])

    points = np.ndarray((len(coordinates), 3))
    for index, coordinate in enumerate(coordinates):
        points[index, :] = coordinate

    n = points.shape[0]
    dist = cdist(points, points).reshape((n * n,))
    dist = dist / np.max(dist)
    number_of_bins = 20
    bins = np.linspace(0, 1, number_of_bins)
    indices = np.digitize(dist, bins)
    distribution = np.bincount(indices)[1:-1]
    norm = np.linalg.norm(distribution)
    distribution = list(distribution) / norm
    return points, distribution


def main():
    seed = 1
    number_of_samples = 20000
    dbd4 = DBD4(size=number_of_samples, ratio=1, thresh=6, seed=seed)
    SecondaryStructureExtractor(dbd4).extract_feature()

    for complex_name in dbd4.complexes:
        print_info(complex_name)
        c = dbd4.complexes[complex_name]
        b_ligand, b_receptor = (c.bound_formation.ligand, c.bound_formation.receptor)
        u_ligand, u_receptor = (c.unbound_formation.ligand, c.unbound_formation.receptor)
        b_ligand_bio_residues, b_receptor_bio_residues = b_ligand.biopython_residues, b_receptor.biopython_residues
        # b_l_ns, b_r_ns = NeighborSearch(b_ligand.atoms), NeighborSearch(b_receptor.atoms)
        u_l_ns, u_r_ns = NeighborSearch(u_ligand.atoms), NeighborSearch(u_receptor.atoms)
        index = randint(1, 30)
        positives = 0
        negatives = 0
        lb2u = c.ligand_bound_to_unbound
        rb2u = c.receptor_bound_to_unbound
        p = False
        n = False
        with PdfPages('{0}/geometry/figures/{1}.pdf'.format(reports_directory, complex_name)) as pdf:
            try:
                for i in range(len(b_ligand_bio_residues)):
                    for j in range(len(b_receptor_bio_residues)):
                        if b_ligand.residues[i] not in lb2u or b_receptor.residues[j] not in rb2u:
                            continue
                        l_atoms = b_ligand_bio_residues[i].get_list()
                        r_atoms = b_receptor_bio_residues[j].get_list()
                        dist_mat = cdist([atom.get_coord() for atom in l_atoms], [atom.get_coord() for atom in r_atoms])
                        if p and n:
                            print "getting out of loop..."
                            raise GetOutOfLoop
                        if dist_mat.min() < dbd4.interaction_threshold:
                            if p:
                                continue
                            positives += 1
                            if positives != index:
                                continue
                            # b_l_points, b_l_dist = get_coords(b_l_ns, b_ligand, b_ligand.residues[i], 0.5, False)
                            # b_r_points, b_r_dist = get_coords(b_r_ns, b_receptor, b_receptor.residues[j], 0.5, False)

                            b_l_points, b_l_dist = get_coords(u_l_ns, u_ligand, lb2u[b_ligand.residues[i]], 0.5, False)
                            b_r_points, b_r_dist = get_coords(u_r_ns, u_receptor, rb2u[b_receptor.residues[j]], 0.5,
                                                              False)

                            u_l_points, u_l_dist = get_coords(u_l_ns, u_ligand, lb2u[b_ligand.residues[i]], 0.5, True)
                            u_r_points, u_r_dist = get_coords(u_r_ns, u_receptor, rb2u[b_receptor.residues[j]], 0.5,
                                                              True)

                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(b_l_points[:, 0], b_l_points[:, 1], b_l_points[:, 2], c='r')
                            ax.scatter(b_r_points[:, 0], b_r_points[:, 1], b_r_points[:, 2], c='b')
                            plt.title("Interacting Residues Bound Conformation")
                            pdf.savefig()
                            plt.close()

                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(u_l_points[:, 0], u_l_points[:, 1], u_l_points[:, 2], c='r')
                            ax.scatter(u_r_points[:, 0], u_r_points[:, 1], u_r_points[:, 2], c='b')
                            plt.title("Interacting Surface Residues Bound Conformation")
                            pdf.savefig()
                            plt.close()

                            plt.figure()
                            plt.plot(u_l_dist)
                            plt.plot(u_r_dist)
                            plt.legend(["bound ligand {0}".format(i), "bound receptor {0}".format(j), "unbound ligand",
                                        "unbound receptor"])
                            pdf.savefig()
                            plt.close()
                            p = True
                        else:
                            if n:
                                continue
                            lb2u = c.ligand_bound_to_unbound
                            rb2u = c.receptor_bound_to_unbound

                            b_l_points, b_l_dist = get_coords(u_l_ns, u_ligand, lb2u[b_ligand.residues[i]], 0.5, False)
                            b_r_points, b_r_dist = get_coords(u_r_ns, u_receptor, rb2u[b_receptor.residues[j]], 0.5,
                                                              False)

                            u_l_points, u_l_dist = get_coords(u_l_ns, u_ligand, lb2u[b_ligand.residues[i]], 0.5, True)
                            u_r_points, u_r_dist = get_coords(u_r_ns, u_receptor, rb2u[b_receptor.residues[j]], 0.5,
                                                              True)

                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(b_l_points[:, 0], b_l_points[:, 1], b_l_points[:, 2], c='r')
                            ax.scatter(b_r_points[:, 0], b_r_points[:, 1], b_r_points[:, 2], c='b')
                            plt.title("Non-Interacting Residues Bound Conformation")
                            pdf.savefig()
                            plt.close()

                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(u_l_points[:, 0], u_l_points[:, 1], u_l_points[:, 2], c='r')
                            ax.scatter(u_r_points[:, 0], u_r_points[:, 1], u_r_points[:, 2], c='b')
                            plt.title("Non-Interacting Surface Residues Bound Conformation")
                            pdf.savefig()
                            plt.close()

                            plt.figure()
                            plt.plot(u_l_dist)
                            plt.plot(u_r_dist)
                            plt.legend(["bound ligand {0}".format(i), "bound receptor {0}".format(j), "unbound ligand",
                                        "unbound receptor"])
                            pdf.savefig()
                            plt.close()
                            n = True

            except GetOutOfLoop:
                pass


if __name__ == "__main__":
    main()