import os
from os.path import splitext
from Bio.PDB import PDBParser, Selection, PPBuilder, CaPPBuilder

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

    # residues = Selection.unfold_entities(structure, 'R')
    atoms = Selection.unfold_entities(structure, 'A')
    polypeptides = PPBuilder().build_peptides(structure)
    if len(polypeptides) == 0:
        polypeptides = CaPPBuilder().build_peptides(structure)
    sequence = ''.join([str(p.get_sequence()) for p in polypeptides])
    residues = [residue for polypeptide in polypeptides for residue in polypeptide]
    protein_name = os.path.basename(file_name).replace(".pdb", "")
    return protein_name, structure, residues, sequence, atoms
    # return protein_name, structure, atoms, residues, polypeptides, sequence
