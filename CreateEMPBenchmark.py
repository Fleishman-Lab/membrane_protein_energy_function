#!/usr/bin/env python3
"""
scripts involved in the calibration of the Elazaridis energy function
"""
import argparse
import os
import sys
import copy
import time
import shutil
import pickle
from collections import OrderedDict
import pandas as pd
from scipy.signal import argrelextrema
from scipy import interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import Rosetta.RosettaFilter as Rf
from utils.Logger import lgr
from MPs.InsertionProfiles import *
from MPs.SplineCalibrationVars import *


mpl.rc_context(fname="%s/.matplotlib/publishable_matplotlibrc" % MY_HOME)
# MY_HOME = os.path.expanduser('~')
# mpl.rc_context(fname="%s/.matplotlib/publishable_matplotlibrc" % MY_HOME)
# LSF_USERNAME = 'jonatha'

# ROSETTA_EXECUTABLES_PATH = '%s/Rosetta/main/source/bin/' % MY_HOME
# ROSETTA_SCRIPTS_EXEC_PATH = 'rosetta_scripts.default.linuxgccrelease'
# ROSETTA_DATABASE_PATH = '%s/Rosetta/main/database/' % MY_HOME
# MEM_POTENTIAL = '%sscoring/score_functions/MembranePotential/' % \
#     ROSETTA_DATABASE_PATH
# PROTOCOLS_PATH = '/home/labs/fleishman/jonathaw/elazaridis/protocols/'
# ELAZAR_POLYVAL_PATH = '%s/membrane_prediciton/' % MY_HOME + \
#     'mother_fucker_MET_LUE_VAL_sym_k_neg.txt'
# MPFilterScan_XML = 'MPFilterScan_ELazaridis.xml'
# MPFilterScanDifferentSFAAS = 'MPFilterScanDifferentSFAAS.xml'

# RMSD_THRESHOLD = 0.5
# NUM_AAS = 26
# POLY_A_NAME = 'polyA'
# NSTRUCT = 1  # 0
# MEMBRANE_HALF_WIDTH = 15
# TOTAL_HALF_WIDTH = 134.5/2  # 50
# LAZARIDIS_POLY_DEG = 4
# FLANK_SIZE = 30  # 6
# TOTAL_AAS = NUM_AAS + (FLANK_SIZE * 2)
# Z = np.linspace(-MEMBRANE_HALF_WIDTH, MEMBRANE_HALF_WIDTH, num=NUM_AAS)
# Z_TOTAL = np.linspace(-TOTAL_HALF_WIDTH, TOTAL_HALF_WIDTH, num=TOTAL_AAS)
# POS_Z_TOT = {i + 1: z for i, z in enumerate(Z_TOTAL)}
# POS_Z_DICT = {i + 1: z for i, z in enumerate(Z)}
# POS_RANGE = range(1, TOTAL_AAS+1)
# AAS_NAMES = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
#              'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL',
#              'TRP', 'TYR']
# AAS_3_1 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
#            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
#            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
#            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
# AAS_1_3 = {v: k for k, v in AAS_3_1.items()}
# AAS = list('ACDEFGHIKLMNPQRSTVWY')
# SKIP_AAS = ['P']  # , 'T', 'S']
# COLOR_MAP = {'full': 'blue', 'no_Menv': 'grey', 'ResSolv': 'purple',
#              'fullCEN': 'blue', 'no_MenvCEN': 'grey', 'ResSolvCEN': 'purple',
#              'elazar': 'red', 'diff_ips': 'orange', 'diff_ips_CEN': 'orange',
#              'fa_intra_rep': 'green', 'fa_mpsolv': 'pink', 'fa_rep': 'black',
#              'p_aa_pp': 'blue', 'rama': 'brown', 'no_res_solv': 'blue',
#              'beta': 'blue', 'beta_no_res_solv': 'black'}
# RMSD_ITERATIONS = {aa: [] for aa in AAS}

# Z_RANGE_AA = {aa: [-20, +20] if aa not in ['R', 'K', 'H'] else [-23, +20] for
#               aa in AAS}

# SPLINE_SMOOTHNESS = 0
# SPLINE_LIM = 25  # 30 1Nov
# PWD = os.getcwd() + '/'


def main():
    """main"""
    global args
    functions = [calibrate_function, calibrate_energy_functions,
                 create_polyA_fasta, sequence_to_idealized_helix,
                 create_spanfile, trunctate_2nd_mem_res,
                 draw_rosetta_profiles_fa_cen, draw_elazar_splines,
                 just_draw_current_profiles, compare_multiple_splines,
                 create_original_ips_table, draw_hbonds_profiles,
                 create_current_elazar_splines_table,
                 just_draw_mpframework_profiles, draw_mpframe_profiles_seaborn,
                 draw_ressolv_profiles_seaborn]
    funcitons = {func.__name__: func for func in functions}

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='', type=str,
                        choices=funcitons.keys())
    parser.add_argument('-queue', default='new-all.q', type=str)
    parser.add_argument('-full', default=False, type=bool)
    parser.add_argument('-show_fig', default=False, type=bool)
    parser.add_argument('-change_rosetta_table', default=False, type=bool)
    parser.add_argument('-improve', default=False, type=bool)
    parser.add_argument('-e_terms', default=['fa_intra_rep', 'fa_mpsolv',
                                             'fa_rep', 'p_aa_pp', 'rama'])
    parser.add_argument('-energy_func_fa', default='talaris2014_elazaridis')
    parser.add_argument('-energy_func_cen')
    parser.add_argument('-note', default=None, type=str,
                        help='add note to spline files')
    parser.add_argument('-use_made_pdb', default=True)
    parser.add_argument('-elec_memb_sig_die', default=False)
    parser.add_argument('-memb_fa_sol', default=False)

    lgr.set_file_name('elazaridis_%s.log' % time.strftime("%H_%M_%d_%b"))

    args = vars(parser.parse_args())

    funcitons[args['mode']](args)

    lgr.close()


def calibrate_energy_functions(args):
    """

    :param args:
    :return:
    """
    global PWD
    # calibrating score5 only, the other score splines are very similar
    fa_cen_for_scores = OrderedDict(dict(score5='centroid',
                                         ref2015='fa_standard'))
    # fa_cen_for_scores = OrderedDict(dict(score4_cart='centroid',
    #                                      ref2015_cart='fa_standard'))
    score_funcs_to_calibrate = list(fa_cen_for_scores.keys())
    original_dir = os.getcwd()
    lgr.log("will calibrate the score functions %r" % score_funcs_to_calibrate)
    for en_func in score_funcs_to_calibrate:
        # if en_func != 'ref2015':
        #     continue
        os.mkdir('%s/%s' % (original_dir, en_func))
        os.chdir('%s/%s' % (original_dir, en_func))
        lgr.create_header("calibrating %s" % en_func)
        PWD = '%s/%s/' % (original_dir, en_func)

        calibrate_function(en_func + '_memb',
                           fa_cen=fa_cen_for_scores[en_func])

        os.chdir('%s/' % original_dir)


def calibrate_function(score_func='talaris2014_elazaridis',
                       fa_cen='fa_standard'):
    """calibrate_function

    :param score_func: name of score function to calibrate
    :param fa_cen: either fa or cen
    """
    # create files for running benchmark
    if args['full']:
        if not args['use_made_pdb']:
            create_polyA_fasta()
            sequence_to_idealized_helix()
        else:
            copy_path = '%s/elazaridis/file_safe/polyA_inMemb.pdb' % MY_HOME
            lgr.log('USING THE PREMADE SAVE PDB !!! from %s' % copy_path)
            shutil.copy(copy_path, 'polyA.pdb')
    # first FilterScan run. using null ResSolv
    full_ips = filterscan_analysis_energy_func(score_func, res_solv_weight=0.0,
                                               fa_cen=fa_cen,
                                               residues_to_test=AAS,
                                               print_xml=True,
                                               adjust_extra_membranal=False)

    lgr.create_header('creating and adjusting elazar profiles')
    elazar_ips = create_elazar_ips()
    elazar_ips = normalize_elazar_profiles(full_ips, elazar_ips)

    # calc the difference InsertionProfiles between Elazar and Rosetta.
    # assign them as the polynom table
    diff_ips = {0: {k: subtract_IP_from_IP(elazar_ips[k], full_ips[k])
                    for k in AAS}}

    create_spline_table(diff_ips[0], 'spline_%s_fa.txt' % score_func,
                        'spline_test_%s.txt' %
                        ('fa' if fa_cen == 'fa_standard' else 'cen'),
                        args['note'])

    # analyse Rosetta again.
    current_ips = {
        0: filterscan_analysis_energy_func(score_func, res_solv_weight=1.0,
                                           fa_cen=fa_cen, residues_to_test=AAS,
                                           adjust_extra_membranal=False,
                                           to_dump_pdbs=False)}
    for aa in AAS:
        rmsd = elazar_ips[aa].rmsd_ips(current_ips[0][aa])
        RMSD_ITERATIONS[aa].append(rmsd)
        lgr.log('the RMSD between Elazar and ResSolv for %s is %.2f' %
                (aa, rmsd))

    # as long as one residue's RMSD is higher than threshold, keep iterating
    if args['improve']:
        args['full'] = True
        rmsds = {aa: elazar_ips[aa].rmsd_ips(current_ips[0][aa]) for
                 aa in AAS if aa not in SKIP_AAS}
        fixed_ips = {}
        iter_num = 1
        while any([rmsd > RMSD_THRESHOLD for rmsd in rmsds.values()]) and \
                iter_num < 10:
            aas_improve = [aa for aa in AAS if aa not in SKIP_AAS
                           if rmsds[aa] > RMSD_THRESHOLD]
            if aas_improve == []:
                lgr.create_header('finished improving')
                break
            lgr.log('starting round %i for AAS %s' % (iter_num, aas_improve))
            diff_ips[iter_num] = {'A': diff_ips[iter_num-1]['A']}

            # check which residues are "good enough" by RMSD, and fix them.
            for aa in AAS:
                if aa in SKIP_AAS:
                    continue
                RMSD_ITERATIONS[aa].append(rmsds[aa])
                if aa not in fixed_ips.keys() and aa not in aas_improve:
                    lgr.log('fixing %s on round %i' % (aa, iter_num))
                    fixed_ips[aa] = diff_ips[iter_num-1][aa]
                if aa not in aas_improve:
                    diff_ips[iter_num][aa] = fixed_ips[aa]

            for aa in aas_improve:
                if aa in SKIP_AAS:
                    continue
                lgr.log('improve %s at %.2f round %i' %
                        (aa, rmsds[aa], iter_num))

                # create a spline that describes the required profile.
                # in the elazar range (-23, 15 or -15, 15)
                # it will be what is required to get to elazar within the
                # membrane. in (inf, -25) and (+25, inf) it is 0.
                y, x = [], []
                for pos in POS_RANGE:
                    if -SPLINE_LIM > POS_Z_TOT[pos] or \
                            POS_Z_TOT[pos] > SPLINE_LIM:
                        y.append(0.0)
                        x.append(pos)
                    # elif elazar_ips[aa].poly_edges[0] <= POS_Z_TOT[pos] <=
                    # elazar_ips[aa].poly_edges[1]:
                    elif elazar_ips[aa].within_edges(POS_Z_TOT[pos]):
                        # train the spline on the difference between the
                        # profile from the previous iteration and what is
                        # reuqired to pull it closer to the Elazar
                        y.append(
                            diff_ips[iter_num-1][aa].pos_score[pos] +
                            elazar_ips[aa].pos_score[pos] -
                            current_ips[iter_num-1][aa].pos_score[pos])
                        x.append(pos)
                tck = interpolate.splrep(x, y, s=SPLINE_SMOOTHNESS)
                diff_ips[iter_num][aa] = InsertionProfile(
                    aa, {pos: interpolate.splev(pos, tck)
                         if -SPLINE_LIM <= POS_Z_TOT[pos] <= +SPLINE_LIM
                         else 0.0 for pos in POS_RANGE})
            create_spline_table(diff_ips[iter_num],
                                'spline_%s_fa_%i.txt' % (score_func, iter_num),
                                'spline_test_%s.txt' %
                                ('fa' if fa_cen == 'fa_standard' else 'cen'),
                                args['note'])
            current_ips[iter_num] = filterscan_analysis_energy_func(
                score_func, res_solv_weight=1.0, fa_cen=fa_cen,
                residues_to_test=AAS, adjust_extra_membranal=False,
                to_dump_pdbs=False)

            rmsds = {aa:
                     elazar_ips[aa].rmsd_ips(
                         current_ips[iter_num][aa])
                     for aa in AAS if aa != 'P'}
            iter_num += 1

    draw_rmsd_plots()
    draw_filterscan_profiles(OrderedDict(
        {'ResSolv': current_ips[iter_num-1],
         'elazar': elazar_ips, 'beta_no_res_solv': full_ips}),
        cen_fa='%s_%s' % (score_func, fa_cen))
    lgr.log('finished calibrating %s %s' % (score_func, fa_cen))
    lgr.log('got these RMSDs:')
    for aa in AAS:
        if aa not in SKIP_AAS:
            lgr.log('for %s got %.2f after %i rounds' %
                    (aa, rmsds[aa], iter_num))


def normalize_elazar_profiles(full_ips: dict, elazar_ips: dict) -> dict:
    """normalize_elazar_profiles

    :param full_ips: {aa: InsertionProfile} for rosetta without MPResSolv
    :type full_ips: dict
    :param elazar_ips: {aa: InsertionProfile} for dsTbL profiles
    :type elazar_ips: dict

    :rtype: dict
    """
    for aa in AAS:
        # for pos in POS_RANGE:
        #     print('*****', aa, full_ips[aa].pos_score[pos])
        rhs_avg = np.mean([full_ips[aa].pos_score[pos]
                           for pos in range(6, 17)])
        lgr.log('for %s found RHS mean to be %.2f' % (aa, rhs_avg))
        for pos in POS_RANGE:
            elazar_ips[aa].pos_score[pos] += rhs_avg
        if aa in 'FVMTSC':
            lgr.log('bounding %s to %.2f' % (aa, rhs_avg))
            for pos in POS_RANGE:
                if aa in 'FVM':
                    if elazar_ips[aa].pos_score[pos] > rhs_avg:
                        elazar_ips[aa].pos_score[pos] = rhs_avg
                if aa in 'T':
                    if elazar_ips[aa].pos_score[pos] < rhs_avg:
                        elazar_ips[aa].pos_score[pos] = rhs_avg
                if aa == 'S':
                    elazar_ips[aa].pos_score[pos] = rhs_avg + 0.84
                if aa == 'C':
                    elazar_ips[aa].pos_score[pos] = rhs_avg + 0.84
    # print('setting the C profile to be the same as S after normalisation')
    # elazar_ips['C'] = copy.deepcopy(elazar_ips['S'])
    return elazar_ips


def draw_rosetta_profiles_fa_cen(args_):
    """draw_rosetta_profiles_fa_cen

    :param args_:
    """
    global PWD
    PWD = os.getcwd()+'/'
    # create files for running benchmark
    if args_['full']:
        if args_['use_made_pdb']:
            copy_path = '%selazaridis/file_safe/polyA_inMemb.pdb' % MY_HOME
            lgr.log('USING THE PREMADE SAVE PDB !!! from %s' % copy_path)
            shutil.copy(copy_path, 'polyA.pdb')
        else:
            create_polyA_fasta()
            sequence_to_idealized_helix()
            create_spanfile()
            trunctate_2nd_mem_res()

    # caluclate and draw full-atom level terms
    fa_e_term_ips, fa_terms = create_e_term_specific_profiles(
        args_, './', args_['energy_func_fa'])

    print('GOT HESES TERMS', fa_terms)
    for term in fa_terms:
        plt.figure()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.15, hspace=0.45)
        for i, aa in enumerate(AAS):
            plt.subplot(5, 4, 1 + i)
            plt.plot(Z_TOTAL, [fa_e_term_ips[term][aa].pos_score[pos]
                               for pos in POS_RANGE], color='k', label=term)
            plt.title(aa.upper())
        plt.suptitle(term)
        plt.savefig('fa_%s.png' % term)
        plt.close()


def create_e_term_specific_profiles(args, path_pdbs: str, energy_func,
                                    res_solv_weight: float=0.0) -> dict:
    """create_e_term_specific_profiles
    creates residue specific insertion profiles for every e_term in the list
    :param args: run arguments
    :param path_pdbs: path to pdbs
    :type path_pdbs: str
    :param energy_func: which energy func to examine
    :param res_solv_weight: weight of res_solv
    :type res_solv_weight: float

    :rtype: dict
    """
    global PWD
    PWD = os.getcwd()+'/'
    if args['full']:
        if args['use_made_pdb']:
            copy_path = '%s/elazaridis/file_safe/polyA_inMemb.pdb' % MY_HOME
            lgr.log('USING THE PREMADE SAVE PDB !!! from %s' % copy_path)
            shutil.copy(copy_path, 'polyA.pdb')
        else:
            create_polyA_fasta()
            sequence_to_idealized_helix()
            create_spanfile()
            trunctate_2nd_mem_res()
        filterscan_analysis_energy_func('ref2015',
                                        res_solv_weight=0.0,
                                        residues_to_test=AAS,
                                        to_dump_pdbs=True,
                                        fa_cen='fa_standard')
    # get energy terms for all res / position combinations
    all_results, e_terms = get_all_e_terms(path_pdbs)

    # create DataFrame
    df_ = pd.DataFrame()
    for aa in AAS:
        for pos in range(1, TOTAL_AAS + 1):
            dct = {'pos': pos, 'aa': aa}
            for e_term in e_terms:
                # print('a', all_results[aa][pos].keys())
                dct[e_term] = all_results[aa][pos][e_term]
            df_ = df_.append(dct, ignore_index=True)
    # each res / position combination has a row with values of all e_terms

    e_term_ips = {}
    for e_term in e_terms:
        # find Ala mean
        A_mean = df_[df_['aa'] == 'A'][e_term].mean()

        # normalise by Ala mean
        df_['%s_normed' % e_term] = df_[e_term] - A_mean

        # create insertion profiles for e_term
        e_term_ips[e_term] = create_insertion_profiles(
            df_, '%s_normed' % e_term, adjust_extra_membranal=False,
            smooth=res_solv_weight == 1.0)

    return e_term_ips, e_terms


def get_all_e_terms() -> (dict, list):
    """get_all_e_terms
    goes over all residue/position combiantions, and gets the total energy
    terms out of their files

    :rtype: (dict, list)
    """
    result = {a: {} for a in AAS}
    all_terms = []
    for res in sorted(AAS_3_1.keys()):
        for ind in range(1, TOTAL_AAS+1):
            for l in open('polyA.pdbALA%i%s.pdb' % (ind, res), 'r'):
                if 'TOTAL_WTD' in l:
                    s = l.split()
                    result[AAS_3_1[res]][ind] = {s[i].replace(':', ''):
                                                 float(s[i+1])
                                                 for i in range(1, len(s), 2)}
                    if not all_terms:
                        all_terms = [s[i].replace(':', '')
                                     for i in range(1, len(s), 2)]
    return result, all_terms


def create_spline_table(diff_ips_: dict, file_name: str,
                        rosetta_spline_table_name, note=None) -> None:
    """
    :param diff_ips_: {AA: IP}
    :param file_name: local file in which to create the table
    :param rosetta_spline_table_name: destination to which place the table if
    "change_rosetta_table"
    :return: None
    """
    with open(PWD+file_name, 'w+') as fout:
        fout.write('# splines generated on %s\n' %
                   time.strftime("%H_%M_%d_%b"))
        if note is not None:
            fout.write('# NOTE: %s\n' % note)
        for aa in AAS:
            if aa in SKIP_AAS:
                fout.write('%s %s\n' %
                           (aa,
                            InsertionProfile(aa, {}).format_spline_energies()))
            else:
                fout.write('%s %s\n' %
                           (aa, diff_ips_[aa].format_spline_energies()))
    lgr.log('created table at %s' % PWD + file_name)
    if args['change_rosetta_table']:
        shutil.copy(PWD + file_name, MEM_POTENTIAL + rosetta_spline_table_name)
        lgr.log('copied table to %s' % MEM_POTENTIAL +
                rosetta_spline_table_name)


def just_draw_mpframework_profiles(args: dict):
    """just_draw_current_profiles"""
    global PWD
    args['full'] = True
    PWD = os.getcwd()+'/'
    if not args['use_made_pdb']:
        lgr.log('making a new polyA.pdb!!!!!!')
        create_polyA_fasta()
        sequence_to_idealized_helix()
        create_polyA_fasta()
        sequence_to_idealized_helix()
    else:
        copy_path = '%s/elazaridis/file_safe/polyA_inMemb.pdb' % MY_HOME
        lgr.log('USING THE PREMADE SAVE PDB !!! from %s' % copy_path)
        shutil.copy(copy_path, 'polyA.pdb')
    # create_spanfile()
    # trunctate_2nd_mem_res()
    elazar_ips = create_elazar_ips()
    current_ips = filterscan_analysis_energy_func('beta_nov15_elazaridis',
                                                  0.0,
                                                  'fa_standard',
                                                  residues_to_test=AAS,
                                                  adjust_extra_membranal=False,
                                                  to_dump_pdbs=False)
    elazar_ips = normalize_elazar_profiles(current_ips, elazar_ips)
    current_ips = filterscan_analysis_energy_func('mpframework_docking_fa_2015',
                                                  0.0,
                                                  'fa_standard',
                                                  residues_to_test=AAS,
                                                  adjust_extra_membranal=False,
                                                  to_dump_pdbs=False)
    dct = {'elazar': elazar_ips,
           'full': current_ips}
    pickle.dump(dct, open('mpframework_profiles_dict.obj', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    draw_filterscan_profiles(dct, show=True)


def just_draw_current_profiles(args: dict):
    """just_draw_current_profiles"""
    global PWD
    args['full'] = True
    PWD = os.getcwd()+'/'
    if not args['use_made_pdb']:
        lgr.log('making a new polyA.pdb!!!!!!')
        create_polyA_fasta()
        sequence_to_idealized_helix()
        create_polyA_fasta()
        sequence_to_idealized_helix()
    else:
        copy_path = '%s/elazaridis/file_safe/polyA_inMemb.pdb' % MY_HOME
        lgr.log('USING THE PREMADE SAVE PDB !!! from %s' % copy_path)
        shutil.copy(copy_path, 'polyA.pdb')
    # create_spanfile()
    # trunctate_2nd_mem_res()
    elazar_ips = create_elazar_ips()
    current_ips = filterscan_analysis_energy_func('ref2015_cart',
                                                  0.0,
                                                  'fa_standard',
                                                  residues_to_test=AAS,
                                                  adjust_extra_membranal=False,
                                                  to_dump_pdbs=False)
    elazar_ips = normalize_elazar_profiles(current_ips, elazar_ips)
    with_resolv = filterscan_analysis_energy_func('ref2015_cart',
                                                  1.0,
                                                  'fa_standard',
                                                  residues_to_test=AAS,
                                                  adjust_extra_membranal=False,
                                                  to_dump_pdbs=False)
    dct = {'elazar': elazar_ips, 'beta': with_resolv,
           'beta_no_res_solv': current_ips}
    import pickle
    pickle.dump(dct, open('dct.obj', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    draw_filterscan_profiles(dct, show=True)


def  draw_ressolv_profiles_seaborn(args: dict):
    path = '/home/labs/fleishman/jonathaw/elazaridis/draw_mpframework_profiles_31Jul'
    beta = pickle.load(open('%s/beta_nov15_elazaridis_no_sol/dct.obj' % path, 'rb'))
    mp = pickle.load(open('%s/mpframework/mpframework_profiles_dict.obj' % path, 'rb'))
    sns.set_style('white')
    mpl.rcParams['axes.linewidth'] = 6  #set the value globally
    plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.15, hspace=0.45)
    print(111, '\n', beta)
    profiles_data = pd.DataFrame()
    for i, aa in enumerate(AAS):
        # ax = plt.subplot(5, 4, 1+i)
        ax = plt.subplot(111)
        # if not aa in ['L']:
        #     continue
        plt.plot(Z_TOTAL, [beta['elazar'][aa].pos_score[pos] for pos in POS_RANGE],
                 label='dsTbL', linewidth=20, alpha=0.6, linestyle='-', c='grey')
        # plt.plot(Z_TOTAL, [beta['beta'][aa].pos_score[pos] for pos in POS_RANGE],
        #          label='beta + residue solvation', linewidth=10, alpha=0.6, linestyle='--',
        #         c='red')
        plt.plot(Z_TOTAL, [beta['beta_no_res_solv'][aa].pos_score[pos] for pos in POS_RANGE],
                 label='beta', linewidth=10, alpha=0.6, linestyle='-',
                c='purple')
        for name, kkk in zip(['dstbl', 'ref2015', 'ref2015_memb'],
                             ['elazar', 'beta_no_res_solv', 'beta']):
            temp = {'aa': aa, 'name': name}
            {pos: beta[kkk][aa].pos_score[pos] for pos in POS_RANGE}
            for pos in POS_RANGE:
                temp[pos] = beta[kkk][aa].pos_score[pos]
            profiles_data = profiles_data.append(temp, ignore_index=True)
        plt.title(aa.upper(), fontsize=70)
        plt.xlim([-15, 15])
        plt.ylim([-6, 8])
        plt.axvline(-15, color='grey')
        plt.axvline(15, color='grey')
        plt.xticks([-15, 0, 15], fontsize=50)
        plt.yticks([-6, -3, 0, 3, 6], fontsize=50)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if aa == 'P':
            plt.yticks([])
        plt.tight_layout()
        legend = plt.legend(loc='best', fontsize=40, frameon=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')
        print('total', Z_TOTAL)
        print('elazar', [beta['elazar']['L'].pos_score[pos] for pos in POS_RANGE])
        print('beta', [beta['beta_no_res_solv'][aa].pos_score[pos] for pos in POS_RANGE])
        print('beta_w', [beta['beta'][aa].pos_score[pos] for pos in POS_RANGE])
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        file_location = '%s/ressolv_%s.pdf' % (path, aa)
        # plt.savefig(file_location, dpi=600)
        plt.close()
    print(profiles_data)
    profiles_data.to_csv('/home/labs/fleishman/jonathaw/plots_general/ref2015_memb_paper_Oct2018/all_profiles.csv')
    # plt.show()


def  draw_mpframe_profiles_seaborn(args: dict):
    path = '/home/labs/fleishman/jonathaw/elazaridis/draw_mpframework_profiles_31Jul'
    beta = pickle.load(open('%s/beta_nov15_elazaridis/dct.obj' % path, 'rb'))
    mp = pickle.load(open('%s/mpframework/mpframework_profiles_dict.obj' % path, 'rb'))
    sns.set_style('white')
    plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.15, hspace=0.45)
    ylims = {'L': [-6, 0], 'G': [0, 7], 'T': [-1, 5]}
    for i, aa in enumerate(AAS):
        ax = plt.subplot(5, 4, 1+i)
        if aa not in ylims.keys():
            continue
        for name, ips in mp.items():
            plt.plot(Z_TOTAL, [ips[aa].pos_score[pos] for pos in POS_RANGE],
                     label=name, linewidth=3, alpha=0.6)
            plt.title(aa.upper())
            plt.xlim([-15, 15])
            plt.ylim(ylims[aa])
            plt.axvline(-15, color='grey')
            plt.axvline(15, color='grey')
            plt.xticks([-15, 0, 15], fontsize=10)
            plt.yticks([-6, -3, 0, 3, 6], fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if aa == 'P':
                plt.yticks([])
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    file_location = '%s/mpframework.pdf' % path
    plt.savefig(file_location, dpi=600)
    # plt.show()


def draw_filterscan_profiles(ips_dict: OrderedDict, cen_fa='fa',
                             show: bool = False) -> None:
    """draw_filterscan_profiles
    draws all profiles
    :param ips_dict: {aa: IP}
    :type ips_dict: OrderedDict
    :param cen_fa: either caetroid or full atom
    :param show: show the fig?
    :type show: bool

    :rtype: None
    """
    plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.15, hspace=0.45)
    for i, aa in enumerate(AAS):
        plt.subplot(5, 4, 1+i)
        for name, ips in ips_dict.items():
            plt.plot(Z_TOTAL, [ips[aa].pos_score[pos] for pos in POS_RANGE],
                     color=COLOR_MAP[name], label=name)
            plt.title(aa.upper())
            plt.xlim([-50, 50])
            plt.axvline(Z_RANGE_AA[aa][0], color='grey')
            plt.axvline(Z_RANGE_AA[aa][1], color='grey')
            plt.axvline(-SPLINE_LIM, color='blue')
            plt.axvline(+SPLINE_LIM, color='blue')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    file_location = '%sprofile_comparison_%s.png' % (PWD, cen_fa)
    plt.savefig(file_location, dpi=600)
    lgr.log('saving profile comparison figure to %s' % file_location)
    if show:
        plt.show()
    else:
        plt.close()


def create_elazar_ips() -> dict:
    """
    creates {AA: ip} for the Elazar scale
    """
    lgr.log('creating InsertionProfiles for Elazar')
    elazar_polyval = MakeHydrophobicityGrade()
    result = {}
    original_ips = {}
    for aa in AAS:
        pos_score = {i+1: a for i, a in enumerate(
            [np.polyval(elazar_polyval[AAS_1_3[aa]], z)
             if -20 <= z <= +20 else 0.0 for z in Z_TOTAL])}

        # extend positive inside rule into the IN side
        if aa in ['R', 'K', 'H']:
            edge_score = np.polyval(elazar_polyval[AAS_1_3[aa]], -20)
            lgr.log('adjust %s -23 <= z <= -20 to be %.2f' % (aa, edge_score))
            for pos in POS_RANGE:
                if -23 <= POS_Z_TOT[pos] <= -20:
                    pos_score[pos] = edge_score

        # force DEQN profiles to be linear, equivalent to membrane core energy
        if aa in ['D', 'E', 'Q', 'N']:
            core_avg = np.mean([pos_score[pos] for pos in POS_RANGE
                                if -10 <= POS_Z_TOT[pos] <= 10])

            # change from 20 to 15 on 12Sep201 to narrow the polar's effect
            for pos in POS_RANGE:
                if -15 <= POS_Z_TOT[pos] <= 15:
                    pos_score[pos] = core_avg

        # force H to be linear in (0, +20)
        if aa == 'H':
            zero_to_twenty_avg = np.max(
                [pos_score[pos] for pos in POS_RANGE
                 if -MEMBRANE_HALF_WIDTH <= POS_Z_TOT[pos]
                 <= MEMBRANE_HALF_WIDTH])
            for pos in POS_RANGE:
                if 0 <= POS_Z_TOT[pos] <= 20:
                    pos_score[pos] = zero_to_twenty_avg

        # for all other AAS, stop polynom influence at furthest max/min points.
        # this prevents the tiny troffs created by the polynom min points
        if aa not in ['D', 'Q', 'N', 'E', 'H', 'A', 'T', 'G', 'K', 'R']:
            x = np.array([pos_score[pos] for pos in POS_RANGE])
            max_pnts = [a+1 for a in argrelextrema(x, np.greater)[0]]
            min_pnts = [a+1 for a in argrelextrema(x, np.less)[0]]
            # for I, there's a minimum at position 56 which is WRONG, skip it..
            if aa == 'I':
                min_pnts = [min_pnts[0]]
                edge_pnts = [POS_Z_TOT[np.min(max_pnts+min_pnts)],
                             POS_Z_TOT[np.max(max_pnts+min_pnts)]]
            if not any([a > 10 for a in edge_pnts]):
                edge_pnts[1] = Z_RANGE_AA[aa][1]
            if not any([a < -10 for a in edge_pnts]):
                edge_pnts[0] = Z_RANGE_AA[aa][0]
        else:
            edge_pnts = [Z_RANGE_AA[aa][0], Z_RANGE_AA[aa][1]]

        lgr.log('for %s setting the poly edges at %.2f, %.2f' %
                (aa, edge_pnts[0], edge_pnts[1]))

        original_ips[aa] = InsertionProfile(aa, pos_score, edge_pnts)
        # for use only to create profiles in kcal/mol
        # adjust kcal/mol to REUs according to kcal/mol=0.57REU.
        # suggested by "Role of conformational sampling in computing mutation-
        # induced changes in protein structure
        # and stability."
        # pos_score = {k: v/0.57 for k, v in pos_score.items()}
        pos_score = {k: v*2.94 for k, v in pos_score.items()}
        ip = InsertionProfile(aa, pos_score=pos_score, poly_edges=edge_pnts)
        result[aa] = ip
    # changing cys to be the same as ser, 12Sep2017
    print('changing C to S')
    print(result['C'].pos_score)
    print(result['S'].pos_score)
    result['C'] = copy.deepcopy(result['S'])
    print(result['C'].pos_score)

    lgr.log('adjusting kcal/mol to REU by REU=kcal/mol / 0.57')
    if args['mode'] == 'elazar_profiles':
        lgr.log('returning original profiles, in kcal/mol')
        return original_ips
    else:
        return result


def create_original_ips_table(args):
    """create_original_ips_table

    :param args: run arguments
    """
    ips = create_elazar_ips()
    polynoms = {}
    plt.figure()
    i = 0
    pos_in_memb = [pos for pos in POS_RANGE if -15 <= POS_Z_TOT[pos] <= 15]
    for aa, ip in ips.items():
        plt.subplot(5, 4, 1+i)
        polynoms[aa] = np.polyfit([POS_Z_TOT[pos] for pos in pos_in_memb],
                                  [ip.pos_score[pos] for pos in pos_in_memb],
                                  LAZARIDIS_POLY_DEG)
        plt.scatter([POS_Z_TOT[pos] for pos in pos_in_memb],
                    [ip.pos_score[pos] for pos in pos_in_memb])
        plt.plot([POS_Z_TOT[pos] for pos in pos_in_memb],
                 [np.polyval(polynoms[aa], POS_Z_TOT[pos])
                  for pos in pos_in_memb])
        plt.title(aa)
        plt.ylim([-2, 2])
        i += 1
        plt.savefig('original_scatter_and_plot.png')
        plt.show()

    with open('original_ips.txt', 'w+') as fout:
        for aa, pln in polynoms.items():
            fout.write('%s %f %f %f %f %f\n' % (aa, pln[0], pln[1], pln[2],
                                                pln[3], pln[4]))


def create_insertion_profiles(df: pd.DataFrame, column: str,
                              adjust_extra_membranal: bool=True,
                              smooth: bool=False) -> dict:
    """
    creates {AA: InsertionProfile} using energy column in df
    """
    lgr.log('creating InsertionProfiles for %s' % column)
    result = {}
    for aa in AAS:
        pos_score = {i: df[((df['aa'] == aa) &
                            (df['pos'] == i))][column].values[0]
                     for i in POS_RANGE}  # switched for -50 to 50

        ip = InsertionProfile(aa, pos_score=pos_score,
                              adjust_extra_membranal=adjust_extra_membranal)
        result[aa] = ip
    return result


def filterscan_analysis_energy_func(energy_function: str,
                                    res_solv_weight: float, fa_cen: str,
                                    residues_to_test: list=AAS,
                                    to_dump_pdbs: bool=False,
                                    adjust_extra_membranal: bool=True,
                                    print_xml: bool=False) -> dict:
    """filterscan_analysis_energy_func
    run the FilteScan protocol on the ployA and return InsertionProfiles dict
    for energy_function

    :param energy_function: which energy function to calibrate
    :type energy_function: str
    :param res_solv_weight: weight for res_solv e term
    :type res_solv_weight: float
    :param fa_cen: either fa or cen
    :type fa_cen: str
    :param residues_to_test: what residues to test
    :type residues_to_test: list
    :param to_dump_pdbs: whether to dump pdbs
    :type to_dump_pdbs: bool
    :param adjust_extra_membranal: whether to adjust the extra membrane (dont)
    :type adjust_extra_membranal: bool
    :param print_xml: print or dont the xml
    :type print_xml: bool

    :rtype: dict
    """
    span_ins_weight = 1 if 'memb' in energy_function else 0
    # score functions
    if args['full']:  # -unmute core.scoring.membrane.MPResSolvEnergy
        if energy_function is None:
            print('no energy function provided!')
            sys.exit()  # -corrections::beta_nov15 -score::elec_memb_sig_die
        cmd = '%s%s ' % (ROSETTA_EXECUTABLES_PATH, ROSETTA_SCRIPTS_EXEC_PATH)
        cmd += '-database %s ' % ROSETTA_DATABASE_PATH
        cmd += '-parser:protocol %s%s ' % (PROTOCOLS_PATH,
                                           MPFilterScanDifferentSFAAS)
        cmd += '-s %s%s.pdb ' % (PWD, POLY_A_NAME)
        cmd += '-nstruct %i ' % NSTRUCT
        cmd += '-overwrite '
        cmd += '-mp:scoring:hbond '
        cmd += '-mute all '
        cmd += '-ex1 -ex2 -ex3 -ex4 '
        cmd += '-parser:script_vars energy_function=%s ' % energy_function
        cmd += 'residues_to_test=%s ' % ''.join(residues_to_test)
        cmd += 'to_dump=%i ' % (1 if to_dump_pdbs else 0)
        cmd += 'res_solv_weight=%.2f ' % res_solv_weight
        cmd += 'fa_or_cen=%s ' % fa_cen
        cmd += 'span_ins_weight=%.2f ' % span_ins_weight
        if 'beta' in energy_function:
            cmd += ' -corrections::beta_nov15 '
            lgr.log('ADDING CORRECTIONS FOR beta')
        if args['elec_memb_sig_die'] and 'beta' in energy_function:
            cmd += ' -score::elec_memb_sig_die '
            lgr.log('ADD ELEC_MEMB_SIG_DIE')
        if args['memb_fa_sol'] :   #and 'elazaridis' in energy_function:
            cmd += ' -score:memb_fa_sol '
            lgr.log('USING MEMB_FA_SOL')
        lgr.log('running FilterScan for %s, cmd:\n%s' %
                (energy_function, cmd))
        lgr.log_text_file('%s' %
                          str(PROTOCOLS_PATH +
                              MPFilterScanDifferentSFAAS),
                          to_print=print_xml)
        os.system(cmd)

        lgr.log('ran FilterScan for energy function %s. finished at %s' %
                (energy_function, time.strftime("%H:%M_%d%b")))

    # parse both sclog files to {pos: {AA: score}}
    if args['full']:
        shutil.move('temp.sclog', '%s.sclog' % energy_function)
        lgr.log('saved the FilterScan sclog to %s.sclog' % energy_function)
        temp_fs_log = parse_filterscan_log('%s.sclog' % energy_function)

    # create DataFrame
    df = pd.DataFrame()
    for aa in AAS:
        for pos in range(1, TOTAL_AAS+1):
            df = df.append({'pos': pos, 'aa': aa,
                            energy_function: temp_fs_log[pos][aa]},
                           ignore_index=True)

    # calculate polyA scores in both score funcitons
    temp_A_mean = df[df['aa'] == 'A'][energy_function].mean()

    # calcualte delta of every mutant and the polyA for both score functions
    df['%s_normed' % energy_function] = df[energy_function]-temp_A_mean

    lgr.log('mean of A for %s is %f' % (energy_function, temp_A_mean))

    ips = create_insertion_profiles(df, '%s_normed' % energy_function,
                                    adjust_extra_membranal,
                                    smooth=adjust_extra_membranal)
    return ips


def draw_rmsd_plots() -> None:
    """
    draw the rmsd over iteration plots
    """
    i = 0
    plt.figure()
    for aa, rmsd_list in RMSD_ITERATIONS.items():
        plt.subplot(5, 4, 1 + i)
        plt.plot(range(len(rmsd_list)), rmsd_list)
        plt.title(aa)

        i += 1
        plt.savefig('rmsd_plt.png')
        plt.close()


def parse_filterscan_log(file_name) -> dict:
    """
    parse a FilterScan run log file, returns dict {position: {AA: score}}
    """
    result = {i: {aa: None for aa in AAS} for i in range(1, TOTAL_AAS + 1)}
    for l in open(file_name, 'r'):
        s = l.split()
        if len(s) >= 4:
            result[int(s[0])][s[2]] = float(s[3])
    return result


def MakeHydrophobicityGrade():
    """
    :return: returns a dictionary of the polynom values for each residue
    """
    global hydrophobicity_polyval
    hydrophobicity_grade = open(ELAZAR_POLYVAL_PATH, 'r')
    polyval = {}
    for line in hydrophobicity_grade:
        split = line.split()
        polyval[AAS_1_3[split[0]].upper()] = [float(n) for n in split[1:6]]
    lgr.log('making T to be G with max at 1')
    polyval['THR'] = polyval['GLY'].copy()
    polyval['THR'][-1] = 1.0
    hydrophobicity_grade.close()
    return polyval


def create_polyA_fasta() -> None:
    """
    creates a fasta file with a num_As polyA in it
    """
    with open(PWD + POLY_A_NAME + '.fa', 'w+') as fout:
        fout.write('>polyA\n%s\n' %
                   ''.join(['A'] * (NUM_AAS + 2 * FLANK_SIZE)))
        lgr.log('created polyA file at %s.fa with %i As and %i flank size' %
                (PWD + POLY_A_NAME, NUM_AAS, FLANK_SIZE))


def sequence_to_idealized_helix() -> None:
    """
    calls the Rosetta application that turns a sequence into a membrane embedded helix pdb
    """
    cmd = '%s%s -in:file:fasta %s -mute all' % (ROSETTA_EXECUTABLES_PATH,
                                                'helix_from_sequence.default.linuxgccreleas', PWD + POLY_A_NAME + '.fa')
    lgr.log('issuing command\n%s' % cmd)
    os.system(cmd)
    try:
        shutil.move(PWD + 'helix_from_sequence.pdb', PWD + POLY_A_NAME + '.pdb')
    except:
        shutil.move(PWD + 'S_0001.pdb', PWD + POLY_A_NAME + '.pdb')
        lgr.log('created an idealised helix from %s and put it in %s' % (POLY_A_NAME, POLY_A_NAME + '.pdb'))


def create_spanfile() -> None:
    """
    creates a simple spanfile for the polyA pdb. basically all 1-num_As residues are in the membrane
    """
    with open('%s%s.span' % (PWD, POLY_A_NAME), 'w+') as fout:
        fout.write('Rosetta-generated spanfile from SpanningTopology object\n')
        fout.write('%i %i\nantiparallel\nn2c\n\t%i	%i\n' %
                   (FLANK_SIZE + 1, FLANK_SIZE + NUM_AAS, FLANK_SIZE + 1,
                    FLANK_SIZE + NUM_AAS))


def trunctate_2nd_mem_res() -> None:
    """
    truncate the 2nd membrane residue from the helix pdb. for some reason Rosetta doesn't like it
    """
    with open('%s%s.pdb' % (PWD, POLY_A_NAME), 'r') as fin:
        with open('%s%s.tmp' % (PWD, POLY_A_NAME), 'w+') as fout:
            for l in fin.read().split('\n'):
                if 'XXXX' not in l:
                    fout.write(l+'\n')
                    shutil.move('%s%s.tmp' % (PWD, POLY_A_NAME), '%s%s.pdb' %
                                (PWD, POLY_A_NAME))
                    lgr.log('removed the second membrane residue')


def draw_elazar_splines(args) -> None:
    """
    make and draw Elazar splines
    :return:
    """
    profiles = MakeHydrophobicityGrade()
    x = np.arange(-50, +50, 0.1)
    Z = np.arange(-15, 15, 0.1)
    plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.15, hspace=0.45)
    for i, aa in enumerate(list('ACDEFGHIKLMNPQRSTVWY')):
        y = [0] * (35 * 10) + [np.polyval(profiles[AAS_1_3[aa]], z)
                               for z in Z] + [0] * (35 * 10)
        if aa not in ['V', 'M', 'H']:
            tck = interpolate.splrep(x, y, s=25)
        else:
            tck = interpolate.splrep(x, y, s=5)
            xnew = np.arange(-50, 50, 0.05)
            ynew = interpolate.splev(xnew, tck, der=0)
            plt.subplot(5, 4, 1 + i)
            plt.plot(x, y, c='k')
            plt.plot(xnew, ynew, c='r')
            plt.scatter(x, [bspleval(x_, tck[0], tck[1], tck[2])
                            for x_ in x], c='g', marker='+')
            plt.vlines(-15, -2, 3, color='grey', linestyles='dashed')
            plt.vlines(15, -2, 3, color='grey', linestyles='dashed')
            plt.title(aa.upper())
            plt.ylim([-2, 3])
            plt.show()


def bspleval(x, knots, coeffs, order, debug=False):
    """
    adopted from http://scipy.github.io/old-wiki/pages/Numpy_Example_List_With_Doc.html

    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    x : list or ndarray
    The set of points at which to evaluate the spline.
    knots : list or ndarray
    The set of knots used to define the spline.
    coeffs : list of ndarray
    The set of spline coefficients.
    order : int
    The order of the spline.

    Returns
    -------
    y : ndarray
    The value of the spline at each point in x.
    """

    k = order
    t = knots
    m = np.alen(t)
    npts = np.alen(x)
    B = np.zeros((m-1, k+1, npts))

    if debug:
        print('k=%i, m=%i, npts=%i' % (k, m, npts))
        print('t=', t)
        print('coeffs=', coeffs)

    # Create the zero-order B-spline basis functions.
    for i in range(m-1):
        B[i, 0, :] = np.float64(np.logical_and(x >= t[i], x < t[i+1]))

    if (k == 0):
        B[m-2, 0, -1] = 1.0

    # Next iteratively define the higher-order basis functions, working from
    # lower order to higher.
    for j in range(1, k+1):
        for i in range(m-j-1):
            if (t[i+j] - t[i] == 0.0):
                first_term = 0.0
            else:
                first_term = ((x - t[i]) / (t[i+j] - t[i])) * B[i, j-1, :]

            if (t[i+j+1] - t[i+1] == 0.0):
                second_term = 0.0
            else:
                second_term = ((t[i+j+1] - x) /
                               (t[i+j+1] - t[i+1])) * B[i+1, j-1, :]

            B[i, j, :] = first_term + second_term
            B[m-j-2, j, -1] = 1.0

    if debug:
        plt.figure()
        for i in range(m-1):
            plt.plot(x, B[i, k, :])
            plt.title('B-spline basis functions')

    # Evaluate the spline by multiplying the coefficients with the highest-
    # order basis functions
    y = np.zeros(npts)
    for i in range(m-k-1):
        y += coeffs[i] * B[i, k, :]

    if debug:
        plt.figure()
        plt.plot(x, y)
        plt.title('spline curve')
        plt.show()

    return(y)


def compare_multiple_splines():
    """
    compare splines from multiple energy funcs
    :return:
    """
    score_funcs = ['score0', 'score1', 'score2', 'score3', 'score5']
    score_splines = {k: dict() for k in score_funcs}
    for score in score_funcs:
        spline_file = sorted([a for a in os.listdir(score)
                              if 'spline' in a])[0]
        for l in open(score+'/'+spline_file, 'r'):
            s = l.split()
            score_splines[score][s[0]] = [float(a) for a in s[1:]]

    elazar_ips = create_elazar_ips()

    plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.15, hspace=0.45)
    for i, aa in enumerate(AAS):
        plt.subplot(5, 4, 1 + i)
        plt.plot(Z_TOTAL, [elazar_ips[aa].pos_score[pos] for pos in POS_RANGE],
                 label='elazar')
        for sc in score_funcs:
            plt.plot(Z_TOTAL, score_splines[sc][aa], label=sc)
            if aa != 'P':
                plt.ylim([-5, 5])
                plt.title(aa)
                plt.legend()
                plt.savefig('spline_comparisons.png')
                plt.show()


def draw_hbonds_profiles(args):
    if args['full']:
        os.mkdir('hbonds_testing')
        os.chdir('hbonds_testing')
        command = "for i in `seq 1 86`;do ~/bin/fleish_sub_general.sh /home/labs/fleishman/jonathaw/Rosetta/main/source/bin/rosetta_scripts.default.linuxgccrelease -parser:protocol ~/elazaridis/protocols/scan_hbonds.xml -s polyA.pdb -script_vars energy_function=beta_nov15_elazaridis res_num=${i} -mp:scoring:hbond -corrections::beta_nov15 -score:elec_memb_sig_die -score:memb_fa_sol -overwrite -out:prefix ${i}_ ;done"
        lgr.log('issuing command\n%s' % command)
        os.system(command)
        os.system("head -2 1_score.sc|tail -1 > all_score.sc")
        os.system("grep SCORE: *_score.sc|grep -v des >> all_score.sc")
    else:
        sc_df = Rf.score_file2df('all_score.sc')
        zs, scs = [], []
        for d, sc in zip(sc_df['description'].values, sc_df['a_e_res']):
            zs.append(POS_Z_TOT[ int( d.split('_')[0] ) ])
            scs.append(sc)
            plt.scatter(zs, scs)
            plt.savefig('hbonds.png')


def create_current_elazar_splines_table(args):
    fout = open('temp_original_elazar_spline.txt', 'w+')
    args['mode'] = 'elazar_profiles'
    ips = create_elazar_ips()
    for k in AAS:
        print(k)
        fout.write('%s %s\n' %
                   (k,
                    ' '.join([str(ips[k].pos_score[pos])
                              for pos in POS_RANGE])))
        fout.close()


if __name__ == '__main__':
    main()
