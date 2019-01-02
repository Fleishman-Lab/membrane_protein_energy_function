import os
import numpy as np

MY_HOME = os.path.expanduser('~')
LSF_USERNAME = 'jonatha'

ROSETTA_EXECUTABLES_PATH = '%s/Rosetta/main/source/bin/' % MY_HOME
ROSETTA_SCRIPTS_EXEC_PATH = 'rosetta_scripts.default.linuxgccrelease'
ROSETTA_DATABASE_PATH = '%s/Rosetta/main/database/' % MY_HOME
MEM_POTENTIAL = '%sscoring/score_functions/MembranePotential/' % \
    ROSETTA_DATABASE_PATH
PROTOCOLS_PATH = '/home/labs/fleishman/jonathaw/elazaridis/protocols/'
ELAZAR_POLYVAL_PATH = '%s/membrane_prediciton/' % MY_HOME + \
    'mother_fucker_MET_LUE_VAL_sym_k_neg.txt'
MPFilterScan_XML = 'MPFilterScan_ELazaridis.xml'
MPFilterScanDifferentSFAAS = 'MPFilterScanDifferentSFAAs.xml'

RMSD_THRESHOLD = 0.5
NUM_AAS = 26
POLY_A_NAME = 'polyA'
NSTRUCT = 1  # 0
MEMBRANE_HALF_WIDTH = 15
TOTAL_HALF_WIDTH = 134.5/2  # 50
LAZARIDIS_POLY_DEG = 4
FLANK_SIZE = 30  # 6
TOTAL_AAS = NUM_AAS + (FLANK_SIZE * 2)
Z = np.linspace(-MEMBRANE_HALF_WIDTH, MEMBRANE_HALF_WIDTH, num=NUM_AAS)
Z_TOTAL = np.linspace(-TOTAL_HALF_WIDTH, TOTAL_HALF_WIDTH, num=TOTAL_AAS)
POS_Z_TOT = {i + 1: z for i, z in enumerate(Z_TOTAL)}
POS_Z_DICT = {i + 1: z for i, z in enumerate(Z)}
POS_RANGE = range(1, TOTAL_AAS+1)
AAS_NAMES = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
             'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL',
             'TRP', 'TYR']
AAS_3_1 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
           'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
           'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
AAS_1_3 = {v: k for k, v in AAS_3_1.items()}
AAS = list('ACDEFGHIKLMNPQRSTVWY')
SKIP_AAS = ['P']  # , 'T', 'S']
COLOR_MAP = {'full': 'blue', 'no_Menv': 'grey', 'ResSolv': 'purple',
             'fullCEN': 'blue', 'no_MenvCEN': 'grey', 'ResSolvCEN': 'purple',
             'elazar': 'red', 'diff_ips': 'orange', 'diff_ips_CEN': 'orange',
             'fa_intra_rep': 'green', 'fa_mpsolv': 'pink', 'fa_rep': 'black',
             'p_aa_pp': 'blue', 'rama': 'brown', 'no_res_solv': 'blue',
             'beta': 'blue', 'beta_no_res_solv': 'black'}
RMSD_ITERATIONS = {aa: [] for aa in AAS}

Z_RANGE_AA = {aa: [-15, +15] if aa not in ['R', 'K', 'H'] else [-23, +20] for
              aa in AAS}

SPLINE_SMOOTHNESS = 0
SPLINE_LIM = 20  # 12Sep2017 changed from 25 to 20
PWD = os.getcwd() + '/'
