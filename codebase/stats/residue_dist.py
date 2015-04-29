import glob
from os.path import splitext
from Bio.PDB import PDBParser, Selection, NeighborSearch
import numpy as np
from matplotlib.pyplot import plot, show

__author__ = 'basir'


def read_pdb_file(file_name, name=None):
    """
    Extract info from a PDB file
        file_name: path of pdb file
        name: name of the structure (default name of the file without extension)
        return:: (structure,R,polypeptides,sequence,seq_res_dict)

            structure: structure object
            residues: list of residues
            polypeptides: list of polypeptides in the structure
            sequence: combined sequence (for all polypeptides)
            seq_res_dict: Sequence to residues mapping index list, sequence[i] corresponds to
                residues[seq_res_dict[i]]
    """

    if name is None:
        name = splitext(file_name)[0]

    structure = PDBParser().get_structure(name, file_name)

    if len(structure) != 1:
        raise ValueError("Unexpected number of structures in " + name)

    residues = Selection.unfold_entities(structure, 'R')
    atoms = Selection.unfold_entities(structure, 'A')
    # polypeptides = PPBuilder().build_peptides(structure)
    # if len(polypeptides) == 0:
    #     polypeptides = CaPPBuilder().build_peptides(structure)
    # sequence = ''.join([p.get_sequence().tostring() for p in polypeptides])
    # res_dict = dict(zip(residues, range(len(residues))))
    # seq_res_dict = [res_dict[residues] for p in polypeptides for residues in p]

    return structure, atoms, residues
        # , polypeptides, sequence, seq_res_dict


def get_residues_distance_distribution(data, r):
    """
    This function computes the distribution of number of residues in certain radius around residues.

    """
    files = glob.glob(data + "*_b.pdb")
    dist = {}
    files.sort()
    file_counter = 0
    for bound_pbd in files:
        dist[bound_pbd] = []
    l = []
    for bound_pbd in files:
        file_counter += 1
        print "Protein " + str(file_counter) + "/" + str(len(files))
        s, a, r = read_pdb_file(bound_pbd)
        ns = NeighborSearch(a)
        res_counter = 0
        for res in r:
            b = 0 * res.child_list[0].get_coord()
            for atom in res.child_list:
                b += atom.get_coord()
            center = b / len(res.child_list)
            l.append(len(ns.search(center, 100, "R")))
            res_counter += 1
            # print "Residue " + str(res_counter) + "out of " + str(len(r))
    plot(np.bincount(l))
    show()
    print files


if __name__ == "__main__":
    data_folder = "../data/DBD4/"
    radius = 100
    get_residues_distance_distribution(data_folder, radius)
