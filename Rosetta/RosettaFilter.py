import re
import os
import io
import sys
import subprocess
import multiprocessing
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DEF_MEM_CUT = {'a_tms_span_fa': 0.5,
               'a_shape': 0.4,
               'a_sasa': 600,
               'a_ddg': -5,
               'a_res_lipo': -5,
               'a_helicality': 4}

def score2dict(file_name: str, verbose=False) -> dict:
    have_fields = False
    results = {}
    with open(file_name, 'r') as fin:
        cont = fin.read().split('\n')
    for l in cont:
        s = l.split()
        if len(s) < 2:
            continue
        if (s[1] == 'total_score' or s[1] == 'score') and not have_fields:
            fields = {a if a[:2] != 'a_' else a[2:]: i
                      for i, a in enumerate(s) if a != 'rms'}
            try:
                fields['rmsd'] = s.index('a_rms')
                fields.pop('rms')  # remove the a_rms, experimental
            except:
                if verbose:
                    print('No rmsd')
            have_fields = True
        elif (s[0] == 'SCORE:' or 'SCORE' in s[0]) and 'score' not in s[1]:
            # adding 1 because I remove rms
            if len(s) != len(list(fields.keys()))+1 if 'rms' in s else 0:
                continue
            # added due to erregularities in some very large score files...
            if s[-1] == s[-2]:
                continue
            results[s[fields['description']]] = {a: float(s[i])
                                                 for a, i in fields.items()
                                                 if a not in
                                                 ['SCORE:', 'description'] and
                                                 'SCORE' not in a}
            if 'score' not in results[s[fields['description']]].keys():
                results[s[fields['description']]]['score'] = results[s[
                    fields['description']]]['total_score']
            elif 'total_score' not in results[s[fields['description']]].keys():
                results[s[fields['description']]]['total_score'] = results[
                    s[fields['description']]]['score']
            results[s[fields['description']]]['description'] = s[
                fields['description']]
    return results


def score_dict2df(sc_dict: dict) -> pd.DataFrame:
    filters = list(sc_dict.values())[0].keys()
    df = pd.DataFrame(columns=filters, index=sc_dict.keys())
    for k, v in sc_dict.items():
        for k1, v1 in v.items():
            df[k1][k] = v1
    return df


def df2boxplots(sc_df: pd.DataFrame) -> None:
    rows = 5
    cols = (len(sc_df.keys()) / 5) + 1
    for i, flt in enumerate(sc_df):
        if flt in ['description', 'SCORE:']:
            continue
        plt.subplot(rows, cols, i+1)
        plt.boxplot(sc_df[flt].tolist())
        plt.title(flt)
    plt.show()


def score_file2df(score_file: str, names_file=None) -> pd.DataFrame:
    try:
        if 'SEQUENCE' in open(score_file, 'r').readline():
            os.system('grep SCORE: %s > %s_; mv %s_ %s' % (score_file,
                                                           score_file,
                                                           score_file,
                                                           score_file))
        df = pd.read_table(score_file, sep='\s+', low_memory=False)
    except pd.io.common.CParserError:
        """
        if score file has different numbers of columns, spearate and concatenate
        to have a usable data frame.
        """
        print('exception!!!')
        sc_text = open(score_file, 'r').read()
        sc_splt = re.compile('SCORE:     score').split(sc_text)
        sc_pars = ['SCORE:     score' + a for a in sc_splt[1:]]
        df_list = []
        first_df_cols = list(pd.read_table(io.StringIO(sc_pars[0]),
                                           sep='\s+').columns)
        for sc_par in sc_pars:
            temp_df = pd.read_table(io.StringIO(sc_par), sep='\s+')
            df_list.append(temp_df[first_df_cols])
        df = pd.concat(df_list)

    score_column = [col for col in df if 'SCORE:' in col][0]
    for column in df:
        if column not in ['description', score_column]:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna()

    if names_file is not None:
        names_list = [a.rstrip('\n') for a in open(names_file, 'r')]
        df = df[df['description'].isin(names_list)]
    if 'score' not in df.columns:
        df['score'] = df['total_score']
    return df


def zgrep_single_out(out_file_: str) -> None:
    subprocess.run(['zgrep SCORE: %s > %s_temp' % (out_file_, out_file_)],
                   shell=1)


def just_read_score(in_file_: str) -> pd.DataFrame:
    return pd.read_csv(in_file_, sep='\s+', low_memory=False,
                       warn_bad_lines=0, error_bad_lines=0)


def zgrep_score(name_: str, out_files_: List[str]) -> pd.DataFrame:
    pool = multiprocessing.Pool()
    pool.map(zgrep_single_out, out_files_)
    # for out_file in out_files_:
    #     subprocess.run(['zgrep SCORE: %s > %s_temp' % (out_file, out_file)],
    #                    shell=1)
    _df = pd.read_csv('%s_temp' % out_files_[0], sep='\s+', low_memory=False,
                      error_bad_lines=0, warn_bad_lines=0)
    # bar_width = len(out_files_) - 1
    # sys.stdout.write("[%s]" % (" " * bar_width))
    # sys.stdout.flush()
    # sys.stdout.write("\b" * (bar_width + 1))  # return to start of line, after [
    all_dfs = pool.map(just_read_score, ['%s_temp' % a for a in out_files_[1:]])
    for temp_df in all_dfs:
        _df = pd.concat([_df, temp_df], axis=0, ignore_index=1)
    # for out_file in out_files_[1:]:
    #     _df = pd.concat([_df, pd.read_csv('%s_temp' % out_file, sep='\s+',
    #                                       low_memory=False, warn_bad_lines=0,
    #                                       error_bad_lines=0)], axis=1)
    for out_file in out_files_:
        os.remove('%s_temp' % out_file)
    #     sys.stdout.write("-")
    #     sys.stdout.flush()
    # sys.stdout.write("]\n")
    _df.to_csv(name_, sep='\t')
    return _df


def get_best_of_best(sc_df: pd.DataFrame,
                     terms: list=['score', 'a_ddg', 'a_pack'],
                     percentile=10) -> pd.DataFrame:
    sets_dict = {}
    for term in terms:
        if term in ['score', 'a_ddg', 'a_res_solv', 'a_mars', 'span_ins']:
            threshold = np.percentile(sc_df[term], percentile)
            sets_dict[term] = set(
                sc_df[sc_df[term] <= threshold]['description'].values)
            print('for %s found threshold %.2f, %i pass' %
                  (term, threshold, len(sets_dict[term])))
        elif term in ['a_sasa', 'a_pack', 'a_shape']:
            threshold = np.percentile(sc_df[term], 100-percentile)
            sets_dict[term] = set(
                sc_df[sc_df[term] >= threshold]['description'].values)
            print('for %s found threshold %.2f, %i pass' %
                  (term, threshold, len(sets_dict[term])))
    final_set = set.intersection(*sets_dict.values())
    return sc_df[sc_df['description'].isin(final_set)]


def get_term_by_threshold(sc_df: pd.DataFrame, score: str, p: float,
                          term: str, func: str) -> float:
    threshold = np.percentile(sc_df[score], p)
    if func == 'min':
        return sc_df[sc_df[score] <= threshold][term].min()
    elif func == 'mean':
        return sc_df[sc_df[score] <= threshold][term].mean()


def get_best_num_by_term(sc_df: pd.DataFrame,
                         num: int=10,
                         term: str='score') -> pd.DataFrame:
    new_df = sc_df.sort_values(by=term)
    new_df = new_df.head(num)
    return new_df


def get_best_percent_by_term(df_: pd.DataFrame,
                             percent_: float=10,
                             term: str='score',
                             over_under: str='under') -> pd.DataFrame:
    threshold = np.percentile(df_[term], percent_)
    if over_under == 'over':
        return df_[df_[term] > threshold]
    elif over_under == 'under':
        return df_[df_[term] < threshold]


def get_z_score_by_rmsd_percent(sc_df: pd.DataFrame,
                                rmsd_name: str='a_rmsd',
                                rmsd_threshold: float=2) -> (float, float):
    # rmsd_threshold = np.nanpercentile(list(sc_df[rmsd_name]), 10)
    # rmsd_threshold = 2
    e_low = sc_df[sc_df[rmsd_name] <= rmsd_threshold]['score']
    # sc_df = sc_df[sc_df['score'] <= np.min(e_low) + 5]
    e_hi = sc_df[sc_df[rmsd_name] > rmsd_threshold]['score']
    return (np.min(e_low) - np.mean(e_hi)) / np.std(e_hi), rmsd_threshold


def remove_failed(df: pd.DataFrame, term: str, ou: str,
                  threshold: float) -> (pd.DataFrame, str):
    if term == 'a_tms_span_fa' and 'a_tms_span_fa' not in df.columns:
        if 'a_tms_span' in df.columns:
            term = 'a_tms_span'
        else:
            sys.exit('no tms_span score')
    if ou == 'over':
        temp_df = df[df[term] >= threshold]
    else:
        if term == 'total_score':
            temp_df = df[df['score'] <= threshold]
        else:
            temp_df = df[df[term] <= threshold]
    return temp_df, '%s left %i with threshold %.2f' % (term, len(temp_df),
                                                        threshold)


def remove_failed_dict(df: pd.DataFrame,
                       term_thresh: dict) -> (pd.DataFrame, dict):
    message = {}
    for k, v in term_thresh.items():
        if type(v) is not dict:
            temp_v = {'threshold': v,
                      'ou': 'over' if k in ['a_sasa', 'a_pack', 'a_shape',
                                            'a_tms_span_fa', 'a_tms_span',
                                            'a_span_topo'] else 'under'}
        else:
            temp_v = v
        df, msg = remove_failed(df, k, temp_v['ou'], temp_v['threshold'])
        message[k] = msg
    return df, message


def get_rmsds_from_table(pymol_calc_file: str) -> pd.DataFrame:
    t = pd.read_csv(pymol_calc_file, header=None,
                    names=['description', 'a_rmsd'], sep='\s',
                    engine='python')
    return t


def get_best_from_list(df_: pd.DataFrame, names_: List[str],
                       n: int = 10) -> pd.DataFrame:
    cln_names = [a.split('/')[-1].split('.pdb')[0] for a in names_]
    df_temp = df_[df_['description'].isin(cln_names)]
    return get_best_num_by_term(df_temp, n)


def set_score_to_ignore_term(df_: pd.DataFrame, term: str) -> pd.DataFrame:
    df_['score'] = df_['score'] - df_[term]
    return df_


def calc_z_score_for_best(sc_df: pd.DataFrame, rmsd_threshold: float=1,
                          score_over_best: float=20,
                          low_rmsd_minimal_count: int=5) -> float:
    pass_df = sc_df[sc_df['score'] <= 0.0]
    low_rmsd_df = pass_df[pass_df['a_rmsd'] <= rmsd_threshold]
    if len(low_rmsd_df) < low_rmsd_minimal_count:
        return None
    high_rmsd_df = pass_df[pass_df['a_rmsd'] >= rmsd_threshold]

    best_energy = np.min(low_rmsd_df['score'])

    high_rmsd_df = high_rmsd_df[high_rmsd_df['score'] <=
                                (best_energy + score_over_best)]
    mean_energy = high_rmsd_df['score'].mean()
    std_energy = high_rmsd_df['score'].std()
    return (best_energy - mean_energy) / std_energy


def merge_score_files(sc_df_1: pd.DataFrame, sc_df_2: pd.DataFrame,
                      term: str, desc_2_modif=None) -> pd.DataFrame:
    sub_df = sc_df_2[[term, 'description']]
    if desc_2_modif:
        temp_df = pd.DataFrame()
        temp_df[term] = sub_df[term]
        temp_df['description'] = sub_df['description'].map(desc_2_modif)
        sub_df = temp_df
    merged_df = pd.merge(sc_df_1, sub_df, on='description')
    return merged_df
