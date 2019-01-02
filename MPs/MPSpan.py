"""
a class describing membrane spans
"""
from enum import Enum
import numpy as np
from sklearn.metrics import r2_score
import PDB.MyPDB as mp
from seq.AASeq import AASeq
from WinGrade import MakeHydrophobicityGrade


class Orientation(Enum):
    """Orientation"""
    in2out = 1
    out2in = 2
    unknwn = 3

    def __str__(self) -> str:
        return self.name


class Span:
    """Span"""
    def __init__(self,
                 start: int = None,
                 end: int = None,
                 orientation: Orientation = None,
                 span_num: int = None,
                 residues: list = None,
                 chain: str = None):
        self.start = start
        self.end = end
        self.orientation = orientation
        self.span_num = span_num
        self.residues = residues
        self.chain = chain
        self.dir_vec, self.mean_vec = self.span_vector()
        self.dir_vec_r2 = self.span_vec_r2()
        self.angle_with_memb = self.angle_with_membrane_normal()

    def __repr__(self):
        return '%i -> %i %s' % (self.start, self.end, self.orientation)

    def span_vector(self) -> (np.array, np.array, np.array):
        first = True

        for res in self.residues:
            for bb_atom in res.iter_bb():
                pnt = [bb_atom.xyz.x, bb_atom.xyz.y, bb_atom.xyz.z]
                if first:
                    data = np.array([pnt])
                    first = False
                else:
                    data = np.append(data, [pnt], axis=0)

        mean = data.mean(axis=0)
        uu, dd, vv = np.linalg.svd(data - mean)
        return vv[0], mean

    def span_vec_r2(self) -> float:
        bbatoms_coords = []
        for res in self.residues:
            for bb_atom in res.iter_bb():
                bbatoms_coords.append(bb_atom.xyz.as_list())
        bbatoms_coords_mat = np.array(bbatoms_coords)
        bbatoms_coords_mat -= self.mean_vec

        ss_tot = np.sum(np.power(bbatoms_coords_mat, 2))

        distances = []
        origin = np.array([0, 0, 0])
        distances = [np.linalg.norm(np.cross(self.dir_vec-origin, origin-x)) /
                     np.linalg.norm(self.dir_vec-origin)
                     for x in bbatoms_coords_mat]
        ss_res = np.sum(np.power(distances, 2))
        return 1 - (ss_res / ss_tot)

    def span_vector_pdb(self) -> None:
        """span_vector_pdb
        print atom representation of the 1st, last and mean dots of the span's
        vector

        :rtype: None
        """
        pnts = self.dir_vec * np.mgrid[-7.5:7.5:2j][:, np.newaxis]
        pnts += self.mean_vec
        pnt0 = pnts[0]
        pnt1 = pnts[1]
        mean = self.mean_vec
        atom0 = mp.Atom(header='ATOM', serial_num=self.span_num*3-2,
                        name='%i' % self.span_num,
                        res_type_3='T%i' % self.span_num,
                        chain='M', res_seq_num=self.span_num,
                        x=pnt0[0], y=pnt0[1], z=pnt0[2])
        atom1 = mp.Atom(header='ATOM', serial_num=self.span_num*3-1,
                        name='%i' % self.span_num,
                        res_type_3='T%i' % self.span_num,
                        chain='M', res_seq_num=self.span_num,
                        x=pnt1[0], y=pnt1[1], z=pnt1[2])
        atom2 = mp.Atom(header='ATOM', serial_num=self.span_num*3,
                        name='%i' % self.span_num,
                        res_type_3='T%i' % self.span_num,
                        chain='M', res_seq_num=self.span_num,
                        x=mean[0], y=mean[1], z=mean[2])

        print(atom0)
        print(atom1)
        print(atom2)

    def angle_with_membrane_normal(self) -> float:
        """angle_with_membrane_normal
        returns the angle (in degrees) between the span and the vecotr normal,
        (0, 0, 1)

        :rtype: float
        """
        memb_normal = np.array([0, 0, 1])
        return np.degrees(np.arccos(np.clip(np.dot(self.dir_vec, memb_normal),
                                            -1.0, 1.0)))

    def get_AASeq(self) -> AASeq:
        """get_AASeq


        :rtype: AASeq
        """
        seq = ''.join([res.res_type for res in self.residues])
        return AASeq(seq, self.span_num)


def create_span(pdb_: mp.MyPDB, start: int, end: int, ori: Orientation = None,
                span_num: int = None, chain: str = None) -> Span:
    """creat_span

    :param pdb_: the pdb for the entry
    :type pdb_: mp.MyPDB
    :param start: start residue for the span
    :type start: int
    :param end: end residue for the span
    :type end: int
    :param span_num: residue number
    :type span_num: int

    :rtype: Span
    """
    if not ori:
        start_z = pdb_.get_res(start).rep_memb_z()
        end_z = pdb_.get_res(end).rep_memb_z()
        if start_z < -3 and end_z > 3:
            ori = Orientation(1)
        elif start_z > 3 and end_z < -3:
            ori = Orientation(2)
        else:
            print('unclear orientation, %s %i %i' % (pdb_.name, start, end))
            ori = Orientation(3)
    try:
        print('reading span', start, end, ori, span_num, chain)
        return Span(start, end, ori, span_num,
                    [pdb_.get_res(a) for a in range(start, end+1)],
                    chain=chain)
    except:
        print('failed on %s %s %i %i' % (pdb_.name, pdb_.keys(), start, end))


def pymol_selection(spans_: list) -> str:
    """pymol_selection

    :param spans_: list of Span instances
    :type spans_: list

    :rtype: str
    """
    txt = ''
    for i, span in enumerate(spans_):
        txt += 'select TM_%i, resi %i-%i\n' % (i+1, span.start, span.end)
    return txt


def calc_span_dG(span_: Span) -> float:
    """calc_span_dG

    :param span_: a span
    :type span_: Span

    :rtype: float
    """
    polyvals = MakeHydrophobicityGrade()
    total = 0.0
    for res in span_.residues:
        if res.res_type in polyvals.keys():
            total += round(np.polyval(polyvals[res.res_type],
                                      res.rep_memb_z()), 1)
    return total
