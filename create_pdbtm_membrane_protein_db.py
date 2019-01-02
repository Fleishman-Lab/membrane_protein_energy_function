#!/usr/bin/env python3
"""
a script to create and analyse membrane protein data base
"""
import os
import argparse
from glob import glob
import urllib.request
import xml.etree.ElementTree
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import PDB.MyPDB as mp
from PDB.MyPDB_funcs import extract_chain, find_transmembrane_spans, parse_PDB
from MPs.MPSpan import create_span, calc_span_dG
from utils.Logger import lgr

MY_HOME = os.path.expanduser('~')
PDBTM_LOCAL = '%s/elazaridis/mp_db/pdbtm' % MY_HOME
ANALYSIS = '%s/elazaridis/mp_db/analysis_pdbtm' % MY_HOME

AAS = list('ACDEFGHIKLMNPQRSTVWY')

sns.set(color_codes=True)
sns.set(style="white")


def main():
    functions = [download_db, extract_tm_chains, update_pdbtm,
                 compile_spans_dataframe, analyse_all_spans_memb_norm_angle,
                 draw_angles_with_memb_normal_dist, analyse_pickled_df,
                 analyse_polar_interactions, aa_z_dist, analyse_leu_patches]
    functions = {func.__name__: func for func in functions}
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', choices=functions.keys())
    parser.add_argument('-names', type=str,
                        help='file with names of previously downloaded pdbs')
    parser.add_argument('-pdb')
    args = vars(parser.parse_args())

    functions[args['mode']](args)


def analyse_leu_patches(args):
    _df = pd.read_pickle('%s/analysis_df.obj' % ANALYSIS)


def aa_z_dist(args: dict) -> None:
    _df = pd.read_pickle('%s/analysis_df.obj' % ANALYSIS)
    aa_z_lsts = {aa: [] for aa in AAS}
    for spn in _df['Span']:
        for res in spn.residues:
            if res.res_type in aa_z_lsts.keys():
                aa_z_lsts[res.res_type].append(res.rep_memb_z())

    _df = pd.DataFrame()
    bin_size = 1
    bins = np.arange(-20, +20+bin_size, bin_size)
    _df['bin_min'] = bins[:-1]
    _df['bin_max'] = bins[1:]
    _df['bin_mean'] = _df['bin_min'] + 0.5
    for aa in AAS:
        _df.loc[:, '%s_count' % aa] = _df.apply(
            lambda r: len(np.where((aa_z_lsts[aa] > r['bin_min']) &
                          (aa_z_lsts[aa] <= r['bin_max']))[0]), axis=1)
    _df.loc[:, 'total_count'] = _df.apply(lambda r: np.sum([r['%s_count' % aa]
                                                            for aa in AAS]),
                                          axis=1)
    for aa in AAS:
        _df['%s_ratio' % aa] = _df['%s_count' % aa] / _df['total_count']
    print(_df)

    for i, aa in enumerate(AAS):
        plt.subplot(5, 4, i+1)
        sns.regplot(x='bin_mean', y='%s_ratio' % aa, data=_df)
        plt.title(aa.upper())
    plt.show()


def analyse_polar_interactions(args: dict) -> None:
    # polars = list('G')
    polars = list('CGPST')
    _df = pd.read_pickle('%s/analysis_df.obj' % ANALYSIS)
    for pdb in set(_df['pdb']):
        for spn1 in _df[_df['pdb'] == pdb]['Span']:
            for spn2 in _df[_df['pdb'] == pdb]['Span']:
                if spn1 == spn2:
                    continue
                # print(spn1, spn2)
                seq1 = spn1.get_AASeq()
                seq2 = spn2.get_AASeq()
                if any([p in seq1.get_seq() for p in polars]) and any(
                    [p in seq2.get_seq() for p in polars]):
                    # print(seq1, seq2)
                    for i_1, res1 in enumerate(spn1.residues):
                        if res1.res_type in polars:
                            for i_2, res2 in enumerate(spn2.residues):
                                if res2.res_type in polars:
                                    # print(res1.res_type, res2.res_type)
                                    dist = res1.get_rep().distance(res2.get_rep())
                                    if dist < 5:
                                        print(pdb, spn1.chain, res1.res_num,
                                              spn2.chain, res2.res_num, dist,
                                              res1.res_type, res2.res_type)
                                        dir1 = '%s_%s' % (res1.res_type,
                                                          res2.res_type)
                                        dir2 = '%s_%s' % (res2.res_type,
                                                          res1.res_type)
                                        if os.path.exists(
                                            '%s/polar_polar_interactions/%s' % (ANALYSIS, dir1)):
                                            dir = dir1
                                        else:
                                            dir = dir2
                                        file_name = '%s/polar_polar_interactions/%s/%s_%s_%i_%i.pdb' % (ANALYSIS, dir, pdb, spn1.chain, spn1.span_num, spn2.span_num)
                                        with open(file_name, 'w+') as fout:
                                            for res in [res1, res2]:
                                                for atom in res.atoms.values():
                                                    fout.write('%s\n' % str(atom))


def analyse_pickled_df(args: dict) -> None:
    _df = pd.read_pickle('%s/analysis_df.obj' % ANALYSIS)
    # _df['memb_norm_ang'] = _df['Span'].map(lambda x:
    #                                        normalise_anlge(x.angle_with_memb))
    # _df['span_len'] = _df['Span'].map(lambda x: len(x.residues))
    # len_bins = [0] + list(np.arange(14, 32, 2)) + [50]
    # _df['span_len_bin'] = _df['span_len'].map(lambda x:
    #                                           alloacte_len_to_bin(x, len_bins))
    # _df['span_vec_r2'] = _df['Span'].map(lambda x: x.span_vec_r2())
    # _df['total_span_num_pdb'] = _df['pdb'].map(lambda x: sum(_df['pdb'] == x))
    # _df.loc[:, 'total_span_num_chain'] = _df.apply(lambda r: len(
    #     _df[(_df['pdb'] == r['pdb']) & (_df['chain'] == r['chain'])]), axis=1)
    print(_df)
    _df = _df[_df['span_vec_r2'] > 0.9]
    draw_angles_with_memb_normal_dist(_df['memb_norm_ang'], 5)
    # sns.distplot(_df['memb_norm_ang'])
    # sns.distplot(_df['Span'].map(lambda x: x.end - x.start))
    # g = sns.FacetGrid(_df, col='total_span_num_chain', col_wrap=5)
    # g.map(sns.distplot, "memb_norm_ang")
    # g = sns.FacetGrid(_df[_df['span_vec_r2'] < 2.2], col='span_len_bin',
    #                   col_wrap=5,
    #                   col_order=['%i-%i' % (len_bins[i], len_bins[i+1])
    #                              for i in range(0, len(len_bins)-1)])
    # g.map(sns.distplot, "memb_norm_ang")
    # plt.show()


def alloacte_len_to_bin(len_: int, bins_: list) -> str:
    digit = np.digitize(len_, bins_)
    return '%i-%i' % (bins_[digit-1], bins_[digit])


def normalise_anlge(ang: float) -> float:
    if ang < 90:
        return ang
    else:
        return np.abs(ang-180)


def draw_angles_with_memb_normal_dist(angles: list, bin_size) -> None:
    ang_bins = np.arange(0, 90+bin_size, bin_size)
    _df = pd.DataFrame()
    _df['min'] = ang_bins[:-1]
    _df['max'] = ang_bins[1:]
    _df.loc[:, 'bin_mean'] = (_df['max'] + _df['min']) / 2
    _df.loc[:, 'count'] = _df.apply(lambda r:
                                    len(np.where((angles > r['min']) &
                                                 (angles <= r['max']))[0]),
                                    axis=1)
    _df['freq'] = _df['count'].map(lambda x: x/_df['count'].sum())
    _df.ix[_df['freq'] == 0, 'freq'] = np.nan
    _df.loc[:, 'ln_p'] = np.log(_df['freq'])
    # p_thresh = np.percentile(_df.dropna()['count'], 5)
    p_thresh = 0.02 * _df['count'].sum()
    print('the perentile threhsold for angle count is %.2f' % p_thresh)
    print('before threhs drop', _df['freq'].sum())
    _df = _df[_df['count'] > p_thresh]
    print('before drop', _df['freq'].sum())
    model_df = _df.dropna()
    print('after drop', _df['freq'].sum())
    print(_df)

    model = make_pipeline(PolynomialFeatures(2), Ridge())
    model.fit(model_df['bin_mean'].values.reshape(-1, 1),
              model_df['freq'].values.reshape(-1, 1))
    ridge = model.named_steps['ridge']
    coefs = ridge.coef_[0]
    intercept = ridge.intercept_
    eq_str = "%f*x^2 + %f*x + %f + %f" % (coefs[2], coefs[1], coefs[0],
                                          intercept)
    model_df.loc[:, 'model'] = model.predict(
        model_df['bin_mean'].values.reshape(-1, 1))
    model_df.loc[:, 'ln_model'] = model_df['model'].map(lambda x: -np.log(x))

    model_df = model_df.dropna()

    model_df.loc[:, 'ref_correction'] = model_df['bin_mean'].map(
        lambda x: np.log(np.sin(np.deg2rad(x))))
    model_df.loc[:, 'ref_n_model'] = model_df['ln_model'] + \
            model_df['ref_correction']

    sim_mod = make_pipeline(PolynomialFeatures(3), Ridge())
    print(model_df)
    sim_mod.fit(model_df['bin_mean'].values.reshape(-1, 1),
                model_df['ref_n_model'].values.reshape(-1, 1))
    sim_ridge = sim_mod.named_steps['ridge']
    sim_coefs = sim_ridge.coef_[0]
    sim_intercept = sim_ridge.intercept_
    sim_eq_str = "%f*x^3+%f*x^2+%f*x+%f+%f" % (sim_coefs[3], sim_coefs[2],
                                               sim_coefs[1], sim_coefs[0],
                                               sim_intercept)
    model_df.loc[:, 'sim_model'] = sim_mod.predict(
        model_df['bin_mean'].values.reshape(-1, 1))

    print(model_df)
    _, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # print(angles)
    # plt.hist(angles, normed=1)
    # plt.show()
    # sns.distplot(angles, axlabel='angle', ax=ax1, hist=1, norm_hist=0,
    #              bins=ang_bins,
    #              hist_kws={'weights': np.ones_like(angles)/float(len(angles))})
    ax1.hist(angles, bins=ang_bins, normed=1,
             weights=np.ones_like(angles)/float(len(angles)))
    ax1.plot(np.arange(0, 90, 0.1),
             np.sin(np.deg2rad(np.arange(0, 90, 0.1)))*(np.pi/180))

    sns.regplot(x='bin_mean', y='freq', fit_reg=1, order=2, data=_df, ax=ax2)

    ax3.plot(model_df['bin_mean'], model_df['ln_model'], label='model')
    ax3.text(0, -10, r'$%s$' % eq_str)
    ax3.text(0, -5, r'$%s$' % sim_eq_str)
    print('eq_str', eq_str)
    print('sim)eq_str', sim_eq_str)
    ax3.plot(model_df['bin_mean'], model_df['ref_correction'], label='ref')
    ax3.plot(model_df['bin_mean'], model_df['ref_n_model'], label='model-ref')
    ax3.plot(model_df['bin_mean'], model_df['sim_model'],
             label='combined_model')
    ax3.legend()

    plt.savefig('3cols.pdf', dpi=400)
    plt.show()
    _, ax4 = plt.subplots(ncols=1)
    hst = sns.distplot(angles, bins=ang_bins, ax=ax4)
    ax4.plot(np.arange(0, 90, 0.1),
                   np.sin(np.deg2rad(np.arange(0, 90, 0.1)))*(np.pi/180))
    hst.axes.set_title('Angle distribution', fontsize=50)
    hst.set_xlabel('Span Angle (degrees)', fontsize=30)
    hst.set_ylabel('Frequency', fontsize=30)
    hst.tick_params(labelsize=20)
    plt.xlim([0, 90])
    plt.savefig('angle_dist.pdf', dpi=400)
    plt.show()
    print(111, sim_eq_str)


def find_span_anlges(name_chain) -> list:
    name, chain = name_chain[0], name_chain[1]
    pdb_file = glob('%s/database/*/%s.trpdb.gz' % (PDBTM_LOCAL, name))
    if not pdb_file:
        # missing.append(name)
        print('missing', name, chain)
        return None
    pdb = parse_PDB(pdb_file[0], name)
    chain_pdb = mp.MyPDB(name=name)
    if chain in pdb.keys():
        chain_pdb.add_chain(pdb.chains[chain])
    else:
        print('pdb %s missing chain %s' % (name, chain))
        return None
    spans = find_spans_by_pdbtm_xml(chain_pdb)
    # spans_my = find_transmembrane_spans(chain_pdb)
    return spans


def find_spans_by_pdbtm_xml(pdb_: mp.MyPDB) -> list:
    chain = list(pdb_.keys())[0]
    xml_file = glob('%s/database/*/%s.xml' % (PDBTM_LOCAL, pdb_.name))
    xml_tree = xml.etree.ElementTree.parse(xml_file[0]).getroot()
    span_lst = []
    for ch in xml_tree.iter('{http://pdbtm.enzim.hu}CHAIN'):
        if ch.get('CHAINID') == chain:
            regions = ch.findall('{http://pdbtm.enzim.hu}REGION')
            for reg in regions:
                if reg.get('type') == 'H':
                    span_lst.append({'start': int(reg.get('pdb_beg')),
                                     'end': int(reg.get('pdb_end'))})
    spans = []
    for i, spn in enumerate(span_lst):
        spans.append(create_span(pdb_, spn['start'], spn['end'], i+1,
                                 chain=chain))
    return spans


def calc_reference_distribution(bins_: np.ndarray) -> list:
    """calc_reference_distribution
    calculate the reference state for span angle with membrane normal.
    :param bins_:
    :type bins_: np.ndarray

    :rtype: list
    """
    total_ = 0
    for bmin, bmax in zip(bins_[::1], bins_[1::1]):
        total_ += np.cos(np.deg2rad(bmin)) - np.cos(np.deg2rad(bmax))
    dist = []
    for bmin, bmax in zip(bins_[::1], bins_[1::1]):
        dist.append((np.cos(np.deg2rad(bmin))-np.cos(np.deg2rad(bmax)))/total_)
    return dist


def analyse_all_spans_memb_norm_angle(args: dict) -> None:
    lgr.set_file_name('all_dataset_angles.log')
    # get a list of all spans in the pdbtm
    pdbtm_names = local_pdbtm_list()
    pool = multiprocessing.Pool()
    spans_list_list = pool.map(find_span_anlges,
                               [(k, v) for k, v in pdbtm_names.items()])
    # for k, v in pdbtm_names.items():
    #     print(k, v)
    #     find_span_anlges([k, v])
    spans = [a for lst in spans_list_list if lst for a in lst if a]
    angles = [span.angle_with_memb for span in spans]

    # normalize angles to 0-180
    angles_norm = []
    for ang in angles:
        if ang < 90:
            angles_norm.append(ang)
        else:
            angles_norm.append(np.abs(ang-180))

    with open('%s/all_dataset_angles.txt' % ANALYSIS, 'w+') as fout:
        fout.write('\n'.join(str(ang) for ang in angles_norm))


def compile_spans_dataframe(args: dict) -> None:
    lgr.set_file_name('all_dataset_angles.log')
    # get a list of all spans in the pdbtm
    pdbtm_names = local_pdbtm_list()
    # i = 1
    # new = OrderedDict()
    # for k, v in pdbtm_names.items():
    #     new.update({k: v})
    #     i += 1
    #     if i == 10:
    #         break
    # pdbtm_names = new
    pool = multiprocessing.Pool(35)
    spans_list_list = pool.map(find_span_anlges,
                               [(k, v) for k, v in pdbtm_names.items()])
    _df = pd.DataFrame()
    pdb_list = [[name] * len(spans_list) for name, spans_list in
                zip(pdbtm_names.keys(), spans_list_list) if spans_list]
    _df['pdb'] = [a for b in pdb_list for a in b]
    _df['chain'] = [pdbtm_names[pdb] for pdb in _df['pdb']]
    _df['Span'] = [spn for lst in spans_list_list if lst for spn in lst]
    _df = _df.dropna(0, 'any')
    _df['seq'] = _df['Span'].map(lambda x: x.get_AASeq().get_seq())
    _df['dG'] = _df['Span'].map(lambda x: calc_span_dG(x))
    _df['memb_norm_ang'] = _df['Span'].map(lambda x:
                                           normalise_anlge(x.angle_with_memb))
    _df['span_len'] = _df['Span'].map(lambda x: len(x.residues))
    len_bins = [0] + list(np.arange(14, 32, 2)) + [50]
    _df['span_len_bin'] = _df['span_len'].map(lambda x:
                                              alloacte_len_to_bin(x, len_bins))
    _df['span_vec_r2'] = _df['Span'].map(lambda x: x.span_vec_r2())
    _df['total_span_num_pdb'] = _df['pdb'].map(lambda x: sum(_df['pdb'] == x))
    _df.loc[:, 'total_span_num_chain'] = _df.apply(lambda r: len(
        _df[(_df['pdb'] == r['pdb']) & (_df['chain'] == r['chain'])]), axis=1)

    _df.to_pickle('%s/analysis_df.obj' % ANALYSIS)


def local_pdbtm_list() -> dict:
    remove = ['4iff_A', '4uwa_A']
    orig = open('%s/pdbtm_alpha_nr.list' % PDBTM_LOCAL, 'r').read().split('\n')
    return OrderedDict([(a.split('_')[0], a.split('_')[1])
                        for a in orig if a and a not in remove])


def extract_tm_chains(args: dict) -> None:
    prev_down = [n.lower()
                 for n in
                 open('%s' % args['names'], 'r').read().split('\n') if n]
    pdbtm_dict = {pdb.split('_')[0]: pdb.split('_')[1] for pdb in prev_down}

    for ent in glob('raw_ents/*/pdb*ent'):
        name = os.path.basename(ent)[3:7]
        extract_chain(
            {'in_file': ent, 'name': name,
             'chains': pdbtm_dict[name],
             'out_file': 'extracted_chains/%s_%s.pdb' %
             (name, pdbtm_dict[name])})


def update_pdbtm(args: dict) -> None:
    """update_pdbtm
    taken from http://pdbtm.enzim.hu/?_=/download/releases
    :param args:
    :type args: dict

    :rtype: None
    """
    os.chdir('~/elazaridis/mp_db/pdbtm')
    os.system('svn update')
    print('if svn returned an error, use svn cleanup inside pdbtm and then ')
    print('svn update to get it to download further')


def get_pdbtm_name_list() -> list:
    pdbtm_nt_path = "http://pdbtm.enzim.hu/data/pdbtm_alpha_nr.list"
    pdbtm_nr_html = urllib.request.urlopen(pdbtm_nt_path)
    names = [name.lower()
             for name in pdbtm_nr_html.read().decode("utf8").split('\n')]
    return names


def download_db(args: dict) -> None:
    from Bio.PDB import PDBList
    os.chdir('raw_ents')
    prev_down = [n.lower()
                 for n in
                 open('../%s' % args['names'], 'r').read().split('\n') if n]
    names = get_pdbtm_name_list()
    pdbl = PDBList()
    errors = []
    downloaded = []
    for name in names:
        try:
            code = name.split('_')[0]
            pdbl.retrieve_pdb_file(name)
            downloaded.append(name)
        except urllib.error.URLError:
            errors.append(code)
            print('failed to retrive %s' % code)
    print('overall failed with %i pdbs' % len(errors))
    print('overall downloaded %i pdbs' % len(downloaded))
    with open(args['names'], 'w+') as fout:
        for name in downloaded + prev_down:
            fout.write('%s\n' % name)
    os.chdir('../')


if __name__ == '__main__':
    main()
