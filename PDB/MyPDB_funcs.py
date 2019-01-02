#!/usr/bin/env python3
"""
functions related to the MyPDB module
"""
import sys
import copy
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List
from biopandas.pdb import PandasPDB
import scipy.linalg as linalg

from utils.general_data import THREE_2_ONE
from seq.AASeq import AASeq, AASeqs
import PDB.MyPDB as mp
from MPs.MPSpan import Span, Orientation


def main():
    """main"""
    functions = [extract_seq, extract_chain, min_distances, ramachadran,
                 interface, change_chain, remove_h]
    functions = {func.__name__ for func in functions}
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', choices=functions.keys())
    parser.add_argument('-in_file')
    parser.add_argument('-out_file', default='tmp.pdb')
    parser.add_argument('-chains', nargs='+', default=['A'])
    parser.add_argument('-name', default=None)
    parser.add_argument('-no_non_residues', default=True, type=bool)
    parser.add_argument('-old_chain')
    parser.add_argument('-new_chain')
    parser.add_argument('-seq_positions', type=int, nargs='+',
                        help='sequence positions to extract', default=False)
    args = vars(parser.parse_args())

    if args['name'] is None:
        args['name'] = args['in_file'].split('.')[0].split('_')[0]

    functions[args['mode']](args)


def extract_seq(args: dict) -> None:
        pdb = parse_PDB(args['in_file'], args['name'])
        if not args['seq_positions']:
            for k, v in pdb.seqs.items():
                if k in args['chains']:
                    print('>%s.%s' % (args['name'], k))
                    print(v.get_seq)
        else:
            for k, v in pdb.seqs.items():
                if k in args['chains']:
                    print('>%s.%s' % (args['name'], k))
                    print('%s' % ''.join(v.get_positions(
                        args['seq_positions'])))


def extract_chain(pdb_: mp.MyPDB, chains_: List[str]) -> None:
    t_pdb = mp.MyPDB(name=pdb_.name)
    for chain in chains_:
        ch_ = pdb_.chains[chain.upper()]
        t_pdb.add_chain(ch_)
    return t_pdb


def min_distances(args: dict) -> None:
    """min_distances

    :param args:
    :type args: dict

    :rtype: None
    """
    pdb = parse_PDB(args['in_file'], args['name'])
    while args['chains']:
        ch_a = args['chains'].pop()
        while args['chains']:
            ch_b = args['chains'].pop()
            print("The minimal distance between chains %s and %s is %f" %
                  (ch_a, ch_b, pdb[ch_a].min_distance_chain(pdb[ch_b])))


def ramachadran(args: dict) -> None:
    """ramachadran

    :param args:
    :type args: dict

    :rtype: None
    """
    pdb = parse_PDB(args['in_file'], args['name'])
    draw_ramachadran(pdb)


def interface(args: dict) -> None:
    """interface

    :param args:
    :type args: dict

    :rtype: None
    """
    pdb = parse_PDB(args['in_file'], args['name'])
    inter_0 = interface_residues(pdb[args['chains'][0]],
                                 pdb[args['chains'][1]])
    inter_1 = interface_residues(pdb[args['chains'][1]],
                                 pdb[args['chains'][0]])
    ch1 = set([a.res_num for a in inter_0])
    ch2 = set([a.res_num for a in inter_1])
    print("select ch_%s_inter, %s and res %s" % (
        args['chains'][0], args['in_file'][:-4],
        '+'.join([str(a) for a in ch1])))
    print("select ch_%s_inter, %s and res %s" %
          (args['chains'][1], args['in_file'][:-4],
           '+'.join([str(a) for a in ch2])))


def change_chain(args: dict) -> None:
    """change_chain

    :param args:
    :type args: dict

    :rtype: None
    """
    pdb = parse_PDB(args['in_file'], args['name'])
    pdb.change_chain_name(args['old_chain'], args['new_chain'])
    write_PDB(args['out_file'], pdb)


def remove_h(args: dict) -> None:
    """remove_h

    :param args:
    :type args: dict

    :rtype: None
    """
    pdb = parse_PDB(args['in_file'], args['name'])
    pdb.remove_hydrogens()
    write_PDB(args['out_file'], pdb)


def find_helix_vector(pdb: mp.MyPDB, start: int, end: int):
    """find_helix_vector

    :param pdb:
    :type pdb: mp.MyPDB
    :param start:
    :type start: int
    :param end:
    :type end: int
    """
    xs, ys, zs = [], [], []
    for i in range(start, end+1):
        res_i = pdb.get_res(i)
        for bb_atom in res_i.iter_bb():
            xs.append(bb_atom.xyz.x)
            ys.append(bb_atom.xyz.y)
            zs.append(bb_atom.xyz.z)
    xs_ = np.array(xs)
    ys_ = np.array(ys)
    zs_ = np.array(zs)

    dist = np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2 + (zs[-1]-zs[0])**2)

    data = np.concatenate((xs_[:, np.newaxis], ys_[:, np.newaxis],
                           zs_[:, np.newaxis]), axis=1)
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    linepts = vv[0] * np.mgrid[-dist/2:dist/2:2j][:, np.newaxis]
    linepts += datamean
    return linepts, data


def find_points_of_closest_distance(l1, l2) -> tuple:
    u = l1[1] - l1[0]
    v = l2[1] - l2[0]
    w = l1[0] - l2[0]
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = (a*c) - (b*b)
    sc, tc = 0.0, 0.0

    if D < 0.000001:
        sc = 0.0
        tc = d / b if b > c else e / c
    else:
        sc = (b*e - c*d) / D
        tc = (a*e - b*d) / D

    # sc = l1[0] + l1*sc
    # tc = l2[0] + l2*tc
    sc = l1[0] + sc * (l1[1] - l1[0])
    tc = l2[0] + tc * (l2[1] - l2[0])
    return sc, tc


def calc_dihedral(p1, p2, p3, p4) -> float:
    p = np.array([p1, p2, p3, p4])
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array([v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0],
                                                                    b[2]]])
    return np.degrees(np.arccos(v[0].dot(v[1])/(np.linalg.norm(v[0]) *
                                                np.linalg.norm(v[1]))))


def translate_and_rotate_res_to_xy_plane(res: mp.Residue,
                                         atom_list: list) -> mp.Residue:
    xyz = copy.deepcopy(res[atom_list[0]].xyz)
    xyz = xyz.scalar_multi(-1)
    res.translate_xyz(xyz)
    # rotate a1 to xy plane
    proj_a2_xy = copy.deepcopy(res[atom_list[1]].xyz)
    proj_a2_xy.z = 0
    a2_copy = copy.deepcopy(res[atom_list[1]].xyz)
    ang_a2_xy = np.arccos(a2_copy.unit().dot(proj_a2_xy.unit()))
    axis_a2_xy = a2_copy.unit().cross(proj_a2_xy.unit()).unit()
    rotation_matrix = rotation_matrix_around_vec(axis_a2_xy, ang_a2_xy)
    res.dot_matrix_me(rotation_matrix)

    # rotate a2 to xy plane
    proj_a2_xy = copy.deepcopy(res[atom_list[2]].xyz)
    proj_a2_xy.z = 0
    a2_copy = copy.deepcopy(res[atom_list[2]].xyz)
    ang_a2_xy = np.arccos(a2_copy.unit().dot(proj_a2_xy.unit()))
    closest_point = point_on_normed_vec_closest_to_point(
        proj_a2_xy.as_nparray(), res[atom_list[1]].xyz.unit().as_nparray())
    ang_a2_xy = angle_between_3_XYZs(a2_copy, closest_point, proj_a2_xy)
    axis = copy.deepcopy(res[atom_list[1]].xyz)
    rotation_matrix = rotation_matrix_around_vec(axis, -ang_a2_xy)
    res.dot_matrix_me(rotation_matrix)

    # rotate so that a2 and 3 are on both sides of the Y axis
    a1_copy = res[atom_list[0]].xyz
    a2_copy = res[atom_list[1]].xyz
    a3_copy = res[atom_list[2]].xyz
    ang_312 = angle_between_3_XYZs(a3_copy, a1_copy, a2_copy)
    ang_y12 = angle_between_3_XYZs(mp.XYZ(0, 1, 0), a1_copy, a2_copy)
    axis = mp.XYZ(0, 0, 1)
    rotation_matrix = rotation_matrix_around_vec(axis,
                                                 - (ang_y12 + 0.5 * ang_312))
    res.dot_matrix_me(rotation_matrix)


def rotation_matrix_around_vec(axis: mp.XYZ, theta: float):
    """
    calcualted the rotation matrix to roatate an boject around axis theta radian
    degrees based on:
    http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/
    and
    http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    return linalg.expm3(np.cross(np.eye(3),
                                 axis.as_list() / np.float64(linalg.norm(
                                     axis.as_list()))*theta))


def write_residues_to_file(residues: list, file_name: str) -> None:
    with open(file_name, 'w+') as fout:
        for res in residues:
            for a in res.atoms.values():
                fout.write('%s\n' % a)


def angle_between_3_XYZs(p1: mp.XYZ, p2: mp.XYZ, p3: mp.XYZ) -> float:
    """
    return the angle between 3 points, in radians:w
    """
    ba = p1.as_nparray() - p2.as_nparray()
    bc = p3.as_nparray() - p2.as_nparray()
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)


def point_on_normed_vec_closest_to_point(p: np.array, v: np.array) -> mp.XYZ:
    vec = v * np.dot(p, v)
    return mp.XYZ(vec[0], vec[1], vec[2])


def parse_membrane_residue(pdb_lines: list) -> mp.MembraneResidue:
    """
    pdb_lines: a list of text lines from .pdb file
    returns a MembraneResidue instance
    """
    result = mp.MembraneResidue()
    for l in pdb_lines:
        s = l.split()
        if s != 0:
            if s[2] == 'THKN':
                result.thkn = mp.XYZ(x=float(s[6]), y=float(s[7]),
                                     z=float(s[8]))
            elif s[2] == 'CNTR':
                result.cntr = mp.XYZ(x=float(s[6]), y=float(s[7]),
                                     z=float(s[8]))
            elif s[2] == 'NORM':
                result.norm = mp.XYZ(x=float(s[6]), y=float(s[7]),
                                     z=float(s[8]))
                result.chain = s[4]
                result.res_num = int(s[5])
    return result


def is_point_infront_points_vec(p1: mp.XYZ, p2: mp.XYZ, x: mp.XYZ) -> float:
    """
    :param p1:point 1 in vec
    :param p2:point 2 in vec
    :param x: a point in space
    :return: True iff point x is "in front" of the direction of vector p2-p1
    >>> p1 = mp.XYZ(0, 0, 0)
    >>> p2 = mp.XYZ(1, 0, 0)
    >>> x = mp.XYZ(1, 1, 0)
    >>> is_point_infront_points_vec(p1, p2, x)
    True
    >>> x = mp.XYZ(-1, -1, 0)
    >>> is_point_infront_points_vec(p1, p2, x)
    False
    """
    oreintation_vec = p2 - p1
    inves_vec = x - p1
    return oreintation_vec.dot(inves_vec) > 0


def extract_seq(pdb: mp.MyPDB) -> dict:
    seqs = {}
    for cid, c in pdb:
        seqs[cid] = AASeq(name='%s.%s' % (pdb.name, cid))
        seq = ''
        for rid, r in c:
            seq += r.res_type
        seqs[cid].set_seq(seq)
    return seqs


def parse_PDB(file_in: str, name: str = None,
              with_non_residue: bool = True) -> mp.MyPDB:
    try:
        pdb = read_pdb_df(file_in, name)
        return pdb
    except:
        print('biopandas failed, using native')
    if file_in[-3:] == '.gz':
        import gzip
        fin = gzip.open(file_in, 'rb')
        cont = fin.read().decode('utf-8').split('\n')
    else:
        fin = open(file_in, 'r')
        cont = fin.read().split('\n')
    pdb = mp.MyPDB(name=name)
    memb_res = []
    for l in cont:
        s = l.split()
        if len(s) < 1:
            continue
        if s[0] in ['ATOM', 'HETATM']:
            if not with_non_residue:
                if l[17:20] not in THREE_2_ONE.keys():
                    continue
            # if 'H' in s[2] and ('1' in s[2] or '2' in s[2] or '3' in s[2]):
                # continue
            if s[3] == 'MEM':
                memb_res.append(l)
                continue
            atom = mp.Atom(header=s[0],
                           serial_num=int(l[6:11]),
                           name=l[12:16].replace(' ', ''),
                           alternate=l[16] if l[16] != ' ' else None,
                           res_type_3=l[17:20],
                           chain=l[21],
                           res_seq_num=int(l[22:26]),
                           achar=l[26],
                           x=float(l[30:38]),
                           y=float(l[38:46]),
                           z=float(l[47:55]),
                           occupancy=float(l[55:61]),
                           temp=float(l[60:66]),
                           si=l[72:76].replace(' ', ''),
                           element=l[76:78].replace(' ', ''),
                           charge=str(l[78:80].replace(' ', '')))
            if atom.alternate is 'B':
                continue
            pdb.add_atom(atom)
    pdb.renumber()
    if memb_res != []:
        mm_res = parse_membrane_residue(memb_res)
        pdb.add_memb_res(mm_res)
    pdb.full_aaseqs = parse_seqres_entry(file_in)
    return pdb


def write_PDB(file_out: str, pdb: mp.MyPDB) -> None:
    atoms = []
    for cid in sorted(pdb.chains.keys()):
        for rid in sorted(pdb[cid].residues.keys()):
            for aid in sorted(pdb[cid][rid].keys()):
                atoms.append(pdb[cid][rid][aid])
    with open(file_out, 'w+') as fout:
        for a in sorted(atoms):
            fout.write(str(a) + '\n')
        if pdb.memb_res:
            fout.write(pdb.memb_res.format_memb_residue())


def draw_ramachadran(pdb: mp.MyPDB) -> None:
    import matplotlib.pyplot as plt
    phis = {}
    psis = {}
    for cid, c in pdb:
        for rid, r in c:
            try:
                prev_res = c[rid - 1]
                phis[rid] = r.phi(prev_res)
            except:
                pass
            try:
                next_res = c[rid + 1]
                psis[rid] = r.psi(next_res)
            except:
                pass

    print('pos phi psi')
    for k, v in phis.items():
        print('%s\t%.2f\t' % (k, v), '%.2f' % psis[k] if k in psis.keys()
              else '')
    plt.scatter(list(phis.values()), list(psis.values()), alpha=0.5)
    plt.xlim((-180., 180))
    plt.ylim((-180., 180))
    plt.xlabel('Phi')
    plt.ylabel('Psi')
    plt.show()


def under_dist(ch1: mp.Chain, ch2: mp.Chain, dist: float=10.0,
               atoms='all') -> list:
    """
    :type ch1: Chain
    :type ch2: Chain
    :param ch1: chain of interest
    :param ch2: other chain in possible interface
    :param dist: maxiaml dist for two residues to be declared as "close"
    :return: a list of Residue instances from ch1 that are close to ch2
    """
    residues = []
    for r1 in ch1.values():
        for r2 in ch2.values():
            if r1 in residues:
                break
            if atoms == 'all':
                if r1.min_distance_res(r2) < dist:
                    residues.append(r1)
            else:
                for atom1 in atoms:
                    for atom2 in atoms:
                        a1 = r1[atom1 if atom1 in r1.keys() else 'CA']
                        a2 = r2[atom2 if atom2 in r2.keys() else 'CA']
                        if a1.distance(a2) < dist:
                            residues.append(r1)
                            break
                    if r1 in residues:
                        break
    return sorted(list(set(residues)), key=lambda x: x.res_num)


def min_dist_neighbor(ch1: mp.Chain, ch2: mp.Chain, dist: int=3) -> list:
    """
    :type ch1: Chain
    :type ch2: Chain
    :param ch1: chain of interest
    :param ch2: other chain in possible interface
    :param dist: number of neigbors from each side to find
    :return: a list of Residue instances from ch1 that are close to ch2
    """
    min_dist, min_dist_res = 10000, None
    for r1 in ch1.values():
        for r2 in ch2.values():
            temp_min = r1.min_distance_res(r2)
            if temp_min < min_dist:
                min_dist = temp_min
                min_dist_res = r1

    residues = []
    for res in ch1.values():
        if min_dist_res.res_num - dist <= res.res_num <= min_dist_res.res_num + dist:
            residues.append(res)
    return sorted(list(set(residues)), key=lambda x: x.res_num)


def interface_residues(ch1: mp.Chain, ch2: mp.Chain, dist: float=10.0) -> list:
    """
    :type ch1: Chain
    :type ch2: Chain
    :param ch1: chain of interest
    :param ch2: other chain in possible interface
    :param dist: maxiaml dist for two residues to be declared as "close"
    :return: a list of Residue instances from ch1 that are close to ch2, point
    to it's direction,
    and do not point to ch1's COM
    """
    residues = []
    ch1_com = ch1.COM()
    for r1 in ch1.values():
        for r2 in ch2.values():
            if r1.min_dist_xyz(r2['CA']) < dist:
                if r1.is_point_infront_of_res(r2['CA'].xyz):
                    if not r1.is_point_infront_of_res(ch1_com):
                        residues.append(r1)
    return list(set(residues))


def com_residues(chain: mp.Chain, residues: list) -> mp.XYZ:
    """
    :param residues: list of residue numbers
    :return: mp.XYZ describing the COM
    """
    Xs, Ys, Zs = [], [], []
    for res in residues:
        resi = chain[res]
        if 'CA' in resi.keys():
            Xs.append(resi['CA'].xyz.x)
            Ys.append(resi['CA'].xyz.y)
            Zs.append(resi['CA'].xyz.z)
    return mp.XYZ(np.mean(Xs), np.mean(Ys), np.mean(Zs))


def memb_residues(pdb: mp.MyPDB) -> list():
    """
    collect a set of residues with memb_z within [-15, 15]
    """
    result = []
    for ch_ in pdb.chains.values():
        for res_ in ch_.values():
            if res_.memb_z is not None:
                result.append(res_)
    return result


def parse_energy_table(in_file: str) -> pd.DataFrame:
    """parse_energy_table

    :param in_file:pdb file name
    :type in_file: str
    :return: data frame of the energy table

    :rtype: pdDataFrame
    """
    i = 0
    for l in open(in_file, 'r'):
        i += 1
        if 'BEGIN_POSE_ENERGIES_TABLE' in l:
            begin_table = i
            continue
        if 'END_POSE_ENERGIES_TABLE' in l:
            end_of_table = i

    df_ = pd.read_table(in_file, header=begin_table, sep=' ',
                        skipfooter=i-end_of_table+1, engine='python')
    df_['res_type_num'] = ['%s_%i' % (a.split('_')[0].split(':')[0],
                                      int(a.split('_')[-1])) if a not in
                           ['weights', 'pose'] else a for a in df_['label']]
    return df_


def read_pdb_df(in_file: str, name: str = None) -> mp.MyPDB:
    """
    parse PDB file using BioPandas, return MyPDB
    """
    ppdb = PandasPDB()
    bp_pdb = ppdb.read_pdb(in_file)
    if name:
        my_pdb = mp.MyPDB(name)
    else:
        my_pdb = mp.MyPDB(name=in_file.split('.')[0])
    memb_atoms = []
    for atom_type in ['ATOM', 'HETATM']:
        for _, row in bp_pdb.df[atom_type].iterrows():
            np_atm = mp.Atom(header=atom_type,
                             serial_num=row['atom_number'],
                             name=row['atom_name'],
                             res_type_3=row['residue_name'],
                             chain=row['chain_id'],
                             res_seq_num=row['residue_number'],
                             x=row['x_coord'],
                             y=row['y_coord'],
                             z=row['z_coord'],
                             occupancy=row['occupancy'],
                             temp=row['b_factor'],
                             element=row['element_symbol'],
                             charge=row['charge'])
            if np_atm.res_type_3 == 'MEM':
                memb_atoms.append(np_atm)
            else:
                my_pdb.add_atom(np_atm)
    if memb_atoms:
        memb_res = mp.MembraneResidue()
        memb_res.set_atoms(memb_atoms)
        my_pdb.add_memb_res(memb_res)
    return my_pdb


def find_transmembrane_spans(pdb_: mp.MyPDB) -> list:
    """find_transmembrane_spans
    returns a list of Span of the pdb
    :param pdb_:
    :type pdb_: mp.MyPDB

    :rtype: list
    """
    spans = []
    start = None
    prev_res = None
    current_ori = None
    start_ori = None
    current_residues = []
    for res in pdb_.iter_all_res():
        # skip if this res is missing backbone atoms
        if not res.bb_complete():
            continue
        in_memb = -15 <= res.rep_memb_z() <= 15

        # determine res orientation
        if res['N'].xyz.z < res['C'].xyz.z:
            current_ori = Orientation(1)
        elif res['N'].xyz.z > res['C'].xyz.z:
            current_ori = Orientation(2)

        # find the next res, if there is one
        next_res = pdb_.get_res(res.res_num+3)
        if next_res and 'N' in next_res.keys() and 'C' in next_res.keys():
            if next_res['N'].xyz.z < next_res['C'].xyz.z:
                next_ori = Orientation(1)
            elif next_res['N'].xyz.z > next_res['C'].xyz.z:
                next_ori = Orientation(2)
        else:
            next_ori = start_ori

        # is this a start of a span?
        if in_memb and not start:
            start = res
            start_ori = current_ori
            current_residues.append(start)

        # not sure what
        elif in_memb and start and start_ori == current_ori:
            current_residues.append(res)

        # found span end
        elif (not in_memb and start) or \
             (in_memb and start_ori != current_ori and next_ori != start_ori):
            if start.rep_memb_z() < -1 and prev_res.rep_memb_z() > 1:
                orientation = Orientation(1)
            elif start.rep_memb_z() > -1 and prev_res.rep_memb_z() < 1:
                orientation = Orientation(2)
            elif prev_res.res_num - start.res_num < 16:
                # print('too short, %i to %i' %
                #       (start.res_num, prev_res.res_num))
                start = None
                current_residues = []
                continue
            else:
                print("unclear span orientation")
                print('starting %i %.2f' % (start.res_num, start.rep_memb_z()))
                print('ending %i %.2f' % (res.res_num, res.rep_memb_z()))
                sys.exit()
            spans.append(Span(start.res_num, prev_res.res_num, orientation,
                              len(spans)+1, current_residues))
            start = None
            current_residues = []
        prev_res = res
    return spans


def print_residues_as_pymol_selection(res_list_: list,
                                      selection_name: str=None,
                                      pdb_name: str=None):
    """print_residues_as_pymol_selection

    :param res_list_: list of residues
    :type res_list_: list
    :param selection_name: optional selection name
    :type selection_name: str
    :param pdb_name: optional pdb name
    :type pdb_name: str
    """
    msg = 'select %s, ' % (selection_name if selection_name else "selection")
    if not pdb_name:
        msg += 'resi '
    for res in res_list_:
        if pdb_name:
            msg += '%s and resi %i, ' % (pdb_name, res.res_num)
        else:
            msg += '%i+' % res.res_num
    if msg[-1] == '+':
        msg = msg[:-1]
    print(msg)


def make_coord_csts(pdb: mp.MyPDB, atoms: List[str]=['CA'], mean: int=0,
                    std: int=1) -> str:
    """make_coord_csts

    :param pdb: MyPDB instance of the pdb
    :type pdb: mp.MyPDB
    :param atoms: a list of atom types to constraint
    :type atoms: List[str]
    :param mean: mean of harmonic dist.
    :type mean: float
    :param std: standard deviation of harmonic dist.
    :type std: float

    :rtype: str
    """
    result = ''
    for res in pdb.iter_all_res():
        if res.res_type == 'MEM':
            continue
        for aid, atm in res:
            if aid in atoms:
                result += 'CoordinateConstraint %s ' % aid
                result += '%i%s ' % (res.res_num, res.chain)
                result += '%s 1A ' % aid
                result += '%.3f %.3f %.3f ' % (atm.xyz.x, atm.xyz.y, atm.xyz.z)
                result += 'HARMONIC %i %i\n' % (mean, std)
    return result


def parse_seqres_entry(pdb_file: str) -> AASeqs:
    aas_3_1 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
               'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
               'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
               'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    results = {}
    for lin in open(pdb_file, 'r'):
        if 'SEQRES' in lin:
            spl = lin.split()
            if spl[2] not in results.keys():
                results[spl[2]] = ''
            for aa in spl[4:]:
                if aa in aas_3_1.keys():
                    results[spl[2]] += aas_3_1[aa]
                else:
                    results[spl[2]] += 'x'
    if not results:
        return None
    aaseqs = AASeqs()
    for chain, seq in results.items():
        aaseqs.add_aaseq(AASeq(seq, chain))
    return aaseqs


def get_general_gpcr_numbering_from_gpcrdb_bfactor(pdb: mp.MyPDB) -> Dict[str, int]:
    """get_general_gpcr_numbering_from_gpcrdb_bfactor

    :param pdb:
    :type pdb: MyPDB

    :rtype: Dictstrint
    >>> pdb = parse_PDB('/home/labs/fleishman/jonathaw/scripts/tests/2vt4_B_0_GPCRDB.pdb')
    >>> dct = get_general_gpcr_numbering_from_gpcrdb_bfactor(pdb)
    >>> dct['B'][40]
    1.31
    >>> dct['B'][342]
    7.52
    >>> dct['B'][237]
    5.68
    """
    result = {ch: {} for ch in pdb.chains.keys()}
    for res in pdb.iter_all_res():
        for atm in res.iter_bb():
            if atm.name in ['CA', 'N']:
                if atm.temp > 0:
                    result[res.chain][atm.res_seq_num] = atm.temp
    return result


def map_res_sequential_num_to_res_num(pdb: mp.MyPDB) -> Dict[int, int]:
    """map_res_sequentioal_num_to_res_num
    map the sequential position of a residue in the PDB sequence to the
    residue number in the pdb

    :param pdb:
    :type pdb: mp.MyPDB

    :rtype: Dictintint
    >>> pdb = parse_PDB('/home/labs/fleishman/jonathaw/scripts/tests/2vt4_B_0_GPCRDB.pdb')
    >>> result = map_res_sequentioal_num_to_res_num(pdb)
    >>> result['B'][39]
    0
    >>> result['B'][342]
    258
    """
    return {ch: {k: chain.seq_pos_from_pdb_pos(res.res_num)
                 for k, res in chain} for ch, chain in pdb}
