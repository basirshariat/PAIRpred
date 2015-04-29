import numpy
import os
import tempfile
from codebase.constants import msms_pdb2xyz, msms_exe, msms_home

__author__ = 'basir'


def get_surface_atoms(pdb_file):
    xyz_file = pdb_file.replace(".pdb", ".xyzr")
    os.chdir(msms_home)
    make_xyz_command = "{0} {1} > {2}".format(msms_pdb2xyz, pdb_file, xyz_file)
    os.system(make_xyz_command)
    assert os.path.isfile(xyz_file), "Failed to generate XYZR file using command:\n".format(make_xyz_command)

    msms_output_file = pdb_file.replace(".pdb", "")
    make_vertex_file_command = "{0} -probe_radius 1.5 -if {1} -of {2} > {3}".format(msms_exe, xyz_file,
                                                                                    msms_output_file,
                                                                                    tempfile.mktemp())
    os.system(make_vertex_file_command)
    surface_file = msms_output_file + ".vert"
    assert os.path.isfile(surface_file), "Failed to generate surface file using command:\n{0}".format(
        make_vertex_file_command)

    with open(surface_file, "r") as file_pointer:
        vertex_list = []
        normal_list = []
        for line in file_pointer.readlines():
            sl = line.split()
            if not len(sl) == 9:
                # skip header
                continue
            vl = [float(x) for x in sl[0:3]]
            nl = [float(x) for x in sl[3:6]]
            vertex_list.append(vl)
            normal_list.append(nl)
    return numpy.array(vertex_list), numpy.array(normal_list)