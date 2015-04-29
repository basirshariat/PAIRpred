# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:50:27 2012
Wrapper for stride (predictor of RASA)
@author: Afsar with modifications from Basir
"""
import numpy as np
import tempfile
import os

from Bio.PDB.Polypeptide import one_to_three

from codebase.constants import amino_acids


to_one_letter_code = {}
aa3idx = {}
for index, amino_acid in enumerate(amino_acids):
    try:
        aa3idx[one_to_three(amino_acid)] = index
        to_one_letter_code[one_to_three(amino_acid)] = amino_acid
    except():
        continue


def get_max_asa(s=None):
    """
    This function returns a dictionary containing the maximum ASA for 
    different residues. when s=single, single letter codes of aa are also
    added to the dictionary
    """
    max_acc = {"ALA": 106.0, "CYS": 135.0, "ASP": 163.0, "GLU": 194.0, "PHE": 197.0, "GLY": 84.0, "HIS": 184.0,
               "ILE": 169.0, "LYS": 205.0, "LEU": 164.0, "MET": 188.0, "ASN": 157.0, "PRO": 136.0, "GLN": 198.0,
               "ARG": 248.0, "SER": 130.0, "THR": 142.0, "VAL": 142.0, "TRP": 227.0, "TYR": 222.0}
    if s is not None and s is 'single':
        for k in max_acc.keys():
            max_acc[to_one_letter_code[k]] = max_acc[k]
    return max_acc


def stride_dict_from_pdb_file(in_file, stride="/usr/bin/stride"):
    """
    Create a Stride dictionary from a PDB file.

    Example:
        stride_dict=stride_dict_from_pdb_file("1fat.pdb")
        (aa,ss,phi,psi,asa,rasa)=stride_dict[('A', 1)]

    @param in_file: pdb file
    @type in_file: string

    @param stride: stride executable (argument to os.system)
    @type stride: string

    @return: a dictionary that maps (chainid, res_id) to
        (aa,ss,phi,psi,asa,rasa)
    @rtype: {}
    #EXample: 
        {('A', '1'): ('GLY', 'C', 360.0, 119.38, 128.2, 1.0),
         ('A', '10'): ('ILE', 'E', -115.8, 136.5, 0.0, 0.0),...}
    Secondary structure codes:
        H	    Alpha helix
        G	    3-10 helix
        I	    PI-helix
        E	    Extended conformation
        B or	b   Isolated bridge
        T	    Turn
        C	    Coil (none of the above)
        IMPORTANT NOTE: if the protein chain	identifier is '	' (space), it
        will	be substituted by '-' (dash) everywhere	in the stride output.
        The same is true  for  command  line	 parameters  involving	chain
        identifiers where you have to specify '-' instead of	' '.
    """
    # import os

    def make_stride_dict(filename):

        """
        Return a stride dictionary that maps (chainid, resname, res_id) to
        aa, ss and accessibility, from a stride output file.
        @param filename: the stride output file
        @type filename: string
        """
        max_acc = get_max_asa()
        stride_out = {}
        handle = open(filename, "r")
        try:
            for l in handle.readlines():
                sl = l.split()
                if sl[0] != "ASG":  # if not detailed secondary structure record
                    continue
                # REM  |---Residue---|    |--Structure--|   |-Phi-|   |-Psi-|  |-Area-|      ~~~~
                #ASG  ALA A    1    1    C          Coil    360.00    -35.26     120.7      ~~~~
                #0      1 2    3    4    5           6       7          8         9          10        
                # In cases where stride cannot recognize the residue type, it puts a '-' there
                # However, Bio.PDB uses ' ' so convert between the two                
                if sl[2] == '-':
                    sl[2] = ' '

                res_id = (sl[2], sl[3])
                aa = sl[1]
                ss = sl[5].upper()  # There was b and B both from Bridge
                phi = float(sl[7])
                psi = float(sl[8])
                asa = float(sl[9])
                try:
                    rasa = asa / max_acc[aa]
                    if rasa > 1.0:  # we do get values greater than 1
                        rasa = 1.0
                except KeyError:
                    rasa = np.nan
                stride_out[res_id] = (aa, ss, phi, psi, asa, rasa)
        finally:
            handle.close()
        return stride_out

    out_file = tempfile.NamedTemporaryFile(suffix='.stride')
    out_file.flush()
    out_file.close()
    temp_file = out_file.name
    os.system("{0} {1} > {2}".format(stride, in_file, temp_file))
    out_dict = make_stride_dict(temp_file)
    os.remove(temp_file)
    return out_dict
