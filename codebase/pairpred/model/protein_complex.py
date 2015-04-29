__author__ = 'basir'

import Bio.pairwise2


class ProteinComplex:
    def __init__(self, complex_name, unbound_formation, bound_formation):
        self.complex_name = complex_name
        self.unbound_formation = unbound_formation
        self.bound_formation = bound_formation
        self.ligand_bound_to_unbound = {}
        self.receptor_bound_to_unbound = {}
        self.map_complex_residues()

    @staticmethod
    def map_protein_residues(bound_protein, unbound_protein):
        """


        """
        bound_to_unbound = {}
        unbound_seq = unbound_protein.sequence
        bound_seq = bound_protein.sequence
        aln = Bio.pairwise2.align.globalxs(unbound_seq, bound_seq, -1, -0.1)
        best_alignment = aln[0]
        unbound_alignment_seq = best_alignment[0]
        bound_alignment_seq = best_alignment[1]
        unbound_index = 0
        bound_index = 0
        for i in range(len(unbound_alignment_seq)):
            if unbound_alignment_seq[i] == bound_alignment_seq[i]:
                bound_to_unbound[bound_protein.residues[bound_index]] = unbound_protein.residues[unbound_index]
            unbound_index += unbound_alignment_seq[i] != '-'
            bound_index += bound_alignment_seq[i] != '-'
        return bound_to_unbound

    def map_complex_residues(self):
        """


        """
        bound_ligand = self.bound_formation.ligand
        unbound_ligand = self.unbound_formation.ligand
        bound_receptor = self.bound_formation.receptor
        unbound_receptor = self.unbound_formation.receptor

        self.ligand_bound_to_unbound = self.map_protein_residues(bound_ligand, unbound_ligand)
        self.receptor_bound_to_unbound = self.map_protein_residues(bound_receptor, unbound_receptor)


