import tempfile
import os
import glob
import numpy as np

from codebase.constants import psaia_home
from codebase.utils.printing import print_error


__author__ = 'basir'


def create_psaia_config_file():
    return 'analyze_bound:\t1\n' \
           'analyze_unbound:\t1\n' \
           'calc_asa:\t1\n' \
           'z_slice:\t0.25\n' \
           'r_solvent:\t1.4\n' \
           'write_asa:\t1\n' \
           'calc_rasa:\t1\n' \
           'standard_asa:\t{0}/amac_data/natural_asa.asa\n' \
           'calc_dpx:\t1\n' \
           'calc_cx:\t1\n' \
           'cx_threshold:\t10\n' \
           'cx_volume:\t20.1\n' \
           'calc_hydro:\t1\n' \
           'hydro_file:\t{0}/amac_data/hydrophobicity.hpb\n' \
           'radii_filename:\t{0}/amac_data/chothia.radii\n' \
           'write_xml:\t0\n' \
           'write_table:\t1\n' \
           'output_dir:\t/tmp'.format(psaia_home)


def run_psaia(pdb_file):
    if psaia_home not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + psaia_home

    input_files_list = tempfile.NamedTemporaryFile(suffix='.psaia')
    input_files_list.write(os.path.abspath(pdb_file))
    input_files_list.flush()

    configuration_file = tempfile.NamedTemporaryFile(suffix='.psaia')
    configuration_file.write(create_psaia_config_file())
    configuration_file.flush()

    command = 'echo "yes" | psa ' + configuration_file.name + ' ' + input_files_list.name
    os.system(command)

    configuration_file.close()
    input_files_list.close()

    protein_name = os.path.splitext(os.path.split(pdb_file)[1])[0]
    psaia_files = glob.glob('/tmp/{0}*bound.tbl'.format(protein_name))
    if len(psaia_files) > 0:
        psaia_file = psaia_files[0]
        psaia_dictionary = make_psaia_dict(psaia_file)
    else:
        psaia_dictionary = None
    for temporary_file in psaia_files:
        os.remove(temporary_file)
    return psaia_dictionary


def make_psaia_dict(filename):
    psaia = {}
    line_number = 0
    try:
        for line in open(filename, "r"):
            line_number += 1
            line_parts = line.split()
            # the line containing 'chain' is the last line before real data starts
            if len(line_parts) < 5 or line_parts[0] == 'chain':
                continue
            protein_id = line_parts[0]
            if protein_id == '*':
                protein_id = ' '
            residue_id = (protein_id, line_parts[6])
            casa = np.array(map(float, line_parts[1:6]))
            rasa = np.array(map(float, line_parts[8:13]))
            rrasa = np.array(map(float, line_parts[13:18]))
            rdpx = np.array(map(float, line_parts[18:24]))
            rcx = np.array(map(float, line_parts[24:30]))
            rhph = np.array(float(line_parts[-1]))
            psaia[residue_id] = (casa, rasa, rrasa, rdpx, rcx, rhph)
    except Exception as e:
        print_error('Error Processing psaia file {0}: {1}'.format(filename, e))
        print_error('Error occurred while processing line: {0}'.format(line_number))
        raise e
    return psaia


if __name__ == "__main__":
    # psaia = run_psaia('../../../../data/DBD4/pdb/1A2K_l_b.pdb')
    # print psaia
    # psaia=make_psaia_dict()
    pass