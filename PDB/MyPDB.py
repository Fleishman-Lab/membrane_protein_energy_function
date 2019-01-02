#!/usr/bin/env python3
import sys
from typing import List
import numpy as np
from collections import OrderedDict
from math import sqrt
from seq.AASeq import AASeq, AASeqs
from utils.general_data import THREE_2_ONE


class XYZ:
    def __init__(self, x: float=None, y: float=None, z: float=None):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return "(%.3f %.3f %.3f)" % (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return "(%f %f %f)" % (self.x, self.y, self.z)

    def __eq__(self, other) -> bool:
        return all([self.x == other.x, self.y == other.y, self.z == other.z])

    def __sub__(self, other):
        """
        :param other:another XYZ instance
        :return:a vector resulting in the subtraction self-other
        >>> a = XYZ(1, 1, 1)
        >>> b = XYZ(2, 2, 2)
        >>> Z = XYZ(-1, -1, -1)
        >>> a-b == Z
        True
        """
        return XYZ(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __abs__(self) -> float:
        """
        :return:the magnitude of XYZ
        >>> a = XYZ(1, 1, 1)
        >>> from math import sqrt
        >>> abs(a) == sqrt(3)
        True
        """
        from math import sqrt
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __add__(self, other):
        return XYZ(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __truediv__(self, num: int):
        return XYZ(self.x / num, self.y / 2, self.z / 2)

    def scalar_multi(self, scalar) -> None:
        return XYZ(self.x * scalar, self.y * scalar, self.z * scalar)

    def cross(self, other):
        """
        :param other: another XYZ instance
        :return:the cross product with self in the left
        >>> x = XYZ(2, 3, 4)
        >>> y = XYZ(5, 6, 7)
        >>> x.cross(y) == XYZ(-3, 6, -3)
        True
        """
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return XYZ(x, y, z)

    def dot(self, other) -> float:
        """
        :param other: another XYZ instance
        :return: the dot product
        >>> a = XYZ(-6, 8, 0)
        >>> b = XYZ(5, 12, 0)
        >>> a.dot(b) == 66
        True
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def unit(self):
        """
        :return: the unit vector
        >>> a = XYZ(-2, 1, 0)
        >>> from math import sqrt
        >>> a.unit() == XYZ(-2/sqrt(5), 1/sqrt(5), 0)
        True
        """
        absi = abs(self)
        return XYZ(self.x / absi, self.y / absi, self.z / absi)

    def distance(self, other) -> float:
        """
        :param other:another atom instance
        :return:the euclidian distance
        >>> a = XYZ(x=1, y=1, z=1)
        >>> b = XYZ(x=1, y=1, z=1)
        >>> a.distance(b)
        0.0
        >>> c = XYZ(x=0, y=0, z=0)
        >>> a.distance(c)
        1.7320508075688772
        """
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 +
                    (self.z - other.z) ** 2)

    def as_list(self) -> list:
        return [self.x, self.y, self.z]

    def as_nparray(self) -> np.array:
        return np.array(self.as_list())

    def is_point_infront_of_points(self, other1, other2) -> bool:
        orientation_vec = other1.as_nparray() - self.as_nparray()
        inves_vec = other2.as_nparray() - self.as_nparray()
        return orientation_vec.dot(inves_vec) > 0


class MembraneResidue:
    def __init__(self):
        self.thkn = Atom()
        self.cntr = Atom()
        self.norm = Atom()
        self.residues = {}
        self.chain = None
        self.res_num = None
        self.res_type = 'MEM'

    def __repr__(self):
        msg = '\tthkn: %r\n' % self.thkn.xyz
        msg += '\tcntr: %r\n' % self.cntr.xyz
        msg += '\tnorm: %r\n' % self.norm.xyz
        return msg

    def __getitem__(self, item):
        if item.lower() == 'thkn':
            return self.thkn
        if item.lower() == 'cntr':
            return self.cntr
        if item.lower() == 'norm':
            return self.norm

    def values(self):
        return [self.thkn, self.cntr, self.norm]

    def keys(self):
        return ['THKN', 'CNTR', 'NORM']

    def format_memb_residue(self) -> str:
        msg = ''
        for atm in [self.thkn, self.cntr, self.norm]:
            msg += '%s\n' % str(atm)
        return msg

    def set_atoms(self, atoms_: List) -> None:
        for atm in atoms_:
            if atm.name == 'THKN':
                self.thkn = atm
            elif atm.name == 'CNTR':
                self.cntr = atm
            elif atm.name == 'NORM':
                self.norm = atm
            else:
                sys.exit('wrong atom added to membrane residue %d' % atm)
        self.chain = atoms_[0].chain
        self.res_num = atoms_[0].res_seq_num


class Atom:
    def __init__(self, header=None, serial_num=None, name=None, alternate='',
                 res_type_3=None, chain=None, res_seq_num=None, x=None, y=None,
                 z=None, achar='', occupancy=1.0, temp=1.0, si='', element='',
                 charge=0):
        self.header = header
        self.serial_num = serial_num
        self.name = name
        self.alternate = alternate
        self.res_type_3 = res_type_3
        self.chain = chain
        self.res_seq_num = res_seq_num
        # self.x = x
        # self.y = y
        # self.z = z
        self.achar = achar
        self.occupancy = occupancy
        self.temp = temp
        self.si = si
        self.element = element
        self.charge = charge
        self.xyz = XYZ(x=x, y=y, z=z)

    def __repr__(self) -> str:
        return self.__str__

    def __str__(self) -> str:
        msg = '%-6s%5d%s%-3s%1s%3s' % (self.header, self.serial_num,
                                       '  ' if len(self.name) < 4 else ' ',
                                       self.name, self.achar, self.res_type_3)
        msg += ' %1s%4d%1s' % (self.chain, self.res_seq_num, self.si)
        msg += '   %8.3f%8.3f%8.3f%6.2f%6.2f' % (self.xyz.x, self.xyz.y,
                                                 self.xyz.z, self.occupancy,
                                                 self.temp)
        msg += '          %2s%2s' % (self.element, self.charge)
        return msg
        # return "%-6s%5d%s%-3s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s" % \
               # (self.header, self.serial_num, '  ' if len(self.name) < 4 else ' ', self.name, self.achar, self.res_type_3, self.chain, self.res_seq_num,
                # self.si, self.xyz.x, self.xyz.y, self.xyz.z, self.occupancy, self.temp, self.element, self.charge)

    def __cmp__(self, other) -> bool:
        return self.serial_num > other.serial_num

    def __gt__(self, other) -> bool:
        return self.serial_num > other.serial_num

    def __eq__(self, other) -> bool:
        return self.serial_num == other.serial_num

    def __ge__(self, other) -> bool:
        return self.serial_num >= other.serial_num

    def dot_me(self, m: np.ndarray):
        vec = np.dot(m, self.xyz.as_list())
        self.xyz = XYZ(vec[0], vec[1], vec[2])

    def set_temp(self, temp: float) -> None:
        """
        B factor
        """
        self.temp = temp

    def set_occupancy(self, occupancy) -> None:
        self.occupancy = occupancy

    def distance(self, other) -> float:
        """
        :param other:another atom instance
        :return:the euclidian distance
        >>> a = Atom(x=1, y=1, z=1)
        >>> b = Atom(x=1, y=1, z=1)
        >>> a.distance(b)
        0.0
        >>> c = Atom(x=0, y=0, z=0)
        >>> a.distance(c)
        1.7320508075688772
        """
        return sqrt((self.xyz.x - other.xyz.x) ** 2 +
                    (self.xyz.y - other.xyz.y) ** 2 +
                    (self.xyz.z - other.xyz.z) ** 2)

    def change_chain_name(self, new: str) -> None:
        self.chain = new

    def translate_xyz(self, xyz: XYZ) -> None:
        """
        :param xyz: XYZ point
        :return: None. translate atom by x, y, z
        """
        self.xyz = self.xyz + xyz

    def set_xyz(self, xyz: XYZ):
        self.xyz = xyz


class Residue:
    def __init__(self, res_type_3: str = None, res_num: int = None,
                 chain: str = None, atoms: dict = None):
        self.res_type_3 = res_type_3
        if res_type_3 in THREE_2_ONE.keys():
            self.res_type = THREE_2_ONE[res_type_3]
        else:
            self.res_type = res_type_3
        self.res_num = res_num
        self.chain = chain
        if atoms is None:
            self.atoms = {}
        else:
            self.atoms = atoms
        self.memb_z = None

    def __repr__(self) -> str:
        if self.memb_z is None:
            msg = 'Chain %s Res #%i Type %s with %i atoms' % (
                self.chain, self.res_num, self.res_type_3, len(self.atoms))
        else:
            msg = 'Chain %s Res #%i Type %s with %i atoms, memb Z %.2f' % (
                self.chain, self.res_num, self.res_type_3, len(self.atoms),
                self.memb_z)
        return msg

    def __getitem__(self, item: str) -> Atom:
        return self.atoms[item]

    def __iter__(self):
        for k, v in self.atoms.items():
            yield k, v

    def iter_bb(self) -> Atom:
        for k, v in self:
            if v.name in ['N', 'CA', 'C', 'O']:
                yield v

    def values(self):
        return self.atoms.values()

    def remove_atom(self, atom: Atom) -> None:
        new_res = {}
        for aid, a in self:
            if a != atom:
                # print(a)
                new_res[aid] = a
        self.atoms = new_res

    def keys(self):
        return self.atoms.keys()

    def add_atom(self, atom: Atom) -> None:
        self.atoms[atom.name] = atom

    def min_distance_res(self, other) -> float:
        """
        :param other:another residue
        :return:the minimum, distance between all atoms in residues
        >>> a = Residue(atoms={1: Atom(x=0, y=0, z=0), 2: Atom(x=1, y=1, z=1)})
        >>> b = Residue(atoms={1: Atom(x=2, y=2, z=2), 2: Atom(x=3, y=3, z=3)})
        >>> a.min_distance_res(b)
        1.7320508075688772
        """
        dists = []
        for mid, m in self:
            for oid, o in other:
                dists.append(m.distance(o))
        return min(dists)

    def phi(self, prev_res) -> float:
        return dihedral(prev_res['C'].xyz, self['N'].xyz, self['CA'].xyz,
                        self['C'].xyz)

    def psi(self, next_res) -> float:
        return dihedral(self['N'].xyz, self['CA'].xyz, self['C'].xyz,
                        next_res['N'].xyz)

    def min_dist_xyz(self, xyz: XYZ):
        if self.res_type != 'G':
            return distance_two_point_to_point(self['CA'].xyz, self['CB'].xyz,
                                               xyz.xyz)
        ha_atoms = [a for a in self.keys() if 'HA' in a]
        mean_pnt = (self[ha_atoms[0]].xyz + self[ha_atoms[1]].xyz) / 2
        return distance_two_point_to_point(self['CA'].xyz, mean_pnt, xyz.xyz)

    def change_chain_name(self, new: str) -> None:
        self.chain = new
        for aid, a in self:
            a.change_chain_name(new)

    def translate_xyz(self, xyz: XYZ) -> None:
        """
        :param xyz: xyz point
        :return: None. translates all residue atoms by x, y, z
        """
        for aid, a in self:
            a.translate_xyz(xyz)

    def dot_matrix_me(self, m: np.ndarray) -> None:
        for a in self.values():
            a.dot_me(m)

    def D_or_L(self) -> str:
        """
        return enantiomer of self, either D or L
        """
        CO = np.array([self['C'].xyz.x, self['C'].xyz.y, self['C'].xyz.z])
        CA = np.array([self['CA'].xyz.x, self['CA'].xyz.y, self['CA'].xyz.z])
        CB = np.array([self['CB'].xyz.x, self['CB'].xyz.y, self['CB'].xyz.z])
        N = np.array([self['N'].xyz.x, self['N'].xyz.y, self['N'].xyz.z])

        v1 = N - CO
        v2 = CA - CO
        cp = np.cross(v1, v2)
        CB_infront = cp.dot(CB-CA) > 0
        print(CB_infront)
        return 'D' if CB_infront else 'L'

    def print_as_pdb(self) -> None:
        for atom in self.atoms.values():
            print(atom)

    def rep_memb_z(self) -> float:
        for atom in ['CA', 'C', 'N', 'O']:
            if atom in self.keys():
                return self[atom].xyz.z
        return self[list(self.keys())[0]].xyz.z

    def get_rep(self) -> Atom:
        for atom in ['CB', 'CA', 'C', 'N', 'O']:
            if atom in self.atoms.keys():
                return self.atoms[atom]

    def bb_complete(self) -> bool:
        """bb_complete
        does the residue have all its bb atoms?
        :rtype: bool
        """
        return all([atom in self.keys() for atom in ['N', 'C', 'CA', 'O']])

    def is_point_infront_of_res(self, pnt: XYZ) -> bool:
        if self.res_type != 'G':
            return self['CA'].xyz.is_point_infront_of_points(self['CB'].xyz,
                                                             pnt)
        ha_atoms = [a for a in self.keys() if 'HA' in a]
        mean_pnt = (self[ha_atoms[0]].xyz + self[ha_atoms[1]].xyz) / 2
        return self['CA'].xyz.is_point_infront_of_points(mean_pnt, pnt)


class Chain:
    def __init__(self, chain_id: str = None, residues: dict = None,
                 non_residues: dict = None):
        self.chain_id = chain_id
        self.residues = residues if residues is not None else OrderedDict()
        if residues is not None:
            self.seq = AASeq(''.join(a.res_type for a in residues.values()))
        else:
            self.seq = AASeq('', name=chain_id)
        self.non_residues = non_residues if non_residues is not None else {}
        if non_residues is not None:
            self.non_residues_seq = AASeq(''.join(a.res_type for a in
                                                  residues.values()),
                                          name=chain_id)
        else:
            self.non_residues_seq = AASeq('', name=chain_id)

    def __repr__(self) -> str:
        return "chain %s has %i residues" % (self.chain_id, len(self.residues))

    def __getitem__(self, item: int) -> Residue:
        try:
            return self.residues[item]
        except:
            return self.non_residues[item]

    def __iter__(self):
        for k, v in self.residues.items():
            yield k, v

    def __len__(self):
        if self.residues == {}:
            return 0
        return len(self.residues.keys())

    def add_residue(self, residue: Residue) -> None:
        if residue.res_type_3 in THREE_2_ONE.keys():
            self.seq.add_aa(residue.res_type)
            self.residues.update({residue.res_num: residue})
        else:
            self.non_residues_seq.add_aa(residue.res_type)
            self.non_residues[residue.res_num] = residue

    def seq_pos_from_pdb_pos(self, seq_pos: int) -> int:
        """seq_pos_from_pdb_pos
        gets a position on the seqeunce of the chain and returns the
        corresponding residue number in PDB numbering

        :param seq_pos: a position on the sequence of the chain. which starts
        at 0 and ends at chain length. not the PDB numbers
        :type seq_pos: int

        :rtype: int
        """
        return list(self.residues.keys()).index(seq_pos)

    def min_distance_chain(self, other: Residue) -> float:
        distances = []
        for mrid, mres in self:
            for orid, ores in other:
                distances.append(mres.min_distance_res(ores))
        return min(distances)

    def keys(self):
        return self.residues.keys()

    def values(self):
        return self.residues.values()

    def COM(self) -> XYZ:
        """
        :return:the Center Of Mass of the chain as calculated by the averages
        over Xs, Ys and Zs of all CAs
        """
        Xs = []
        Ys = []
        Zs = []
        for res in self.values():
            if 'CA' in res.keys():
                Xs.append(res['CA'].xyz.x)
                Ys.append(res['CA'].xyz.y)
                Zs.append(res['CA'].xyz.z)
        return XYZ(np.mean(Xs), np.mean(Ys), np.mean(Zs))

    def change_chain_name(self, new: str) -> None:
        self.chain_id = new
        for rid, r in self:
            r.change_chain_name(new)

    def translate_xyz(self, xyz: XYZ) -> None:
        """
        :param xyz: an xyz point
        :return: None. translate all chain atoms by xyz
        """
        for rid, r in self:
            r.translate_xyz(xyz)


class MyPDB:
    def __init__(self, name: str = None, chains: dict = None,
                 seqs: dict = None):
        self.name = name
        self.chains = chains if chains is not None else {}
        self.seqs = seqs if seqs is not None else {}
        self.memb_res = None
        self.aaseqs = AASeqs(name=self.name)
        self.full_aaseqs = AASeqs(name=self.name+'_full')

    @property
    def __repr__(self) -> str:
        msg = "PDB %s has %i chains " % (self.name, len(self.chains))
        for c in self.chains:
            msg += repr(self.chains[c])
        return msg

    def __str__(self):
        return self.__repr__

    def __getitem__(self, item: str) -> Chain:
        return self.chains[item]

    def __iter__(self):
        for k, v in self.chains.items():
            yield k, v

    def keys(self) -> list:
        return self.chains.keys()

    def seq_length(self):
        return sum([len(v) for k, v in self])

    def iter_all_res(self):
        for ch in sorted(self.chains.keys()):
            for res in sorted(self.chains[ch].residues.keys()):
                if res in self[ch].keys():
                    yield self[ch][res]
                else:
                    yield self[ch].non_residues[res]

    def res_items(self):
        for ch in sorted(self.chains.keys()):
            for id, res in sorted(self.chains[ch].residues.items()):
                yield id, res

    def get_res(self, res_num: int) -> Residue:
        for cid, c in self:
            if res_num in c.keys():
                return c[res_num]
            # else:
            #     return c[res_num]

    def add_chain(self, chain: Chain) -> None:
        """
        :param chain: a Chain
        :return: appends chain to PDB
        """
        self.chains[chain.chain_id] = chain

    def add_atom(self, atom: Atom) -> None:
        if atom.chain not in self.chains.keys():
            self.add_chain(chain=Chain(chain_id=atom.chain))
        if atom.res_seq_num not in self.chains[atom.chain].residues.keys():
            self[atom.chain].add_residue(residue=Residue(
                res_type_3=atom.res_type_3, res_num=atom.res_seq_num,
                chain=atom.chain))
        self[atom.chain][atom.res_seq_num].add_atom(atom=atom)
        if atom.res_type_3 in THREE_2_ONE.keys():
            self.seqs[atom.chain] = self[atom.chain].seq
            self.aaseqs['%s.%s' % (self.name, atom.chain)] = self[atom.chain].seq
        else:
            self.seqs[atom.chain + '_non_res'] = \
                self[atom.chain].non_residues_seq
            self.aaseqs['%s.%s_non_res' %
                        (self.name, atom.chain)] = self[atom.chain].seq

    def change_chain_name(self, old: str, new: str) -> None:
        self.chains[old].change_chain_name(new)

    def renumber(self) -> None:
        """
        :return: renumbers self
        """
        i = 1
        for cid, c in self:
            for rid, r in c:
                for aid, a in r:
                    self[cid][rid][aid].serial_num = i
                    i += 1

    def remove_hydrogens(self) -> None:
        """
        removes all Hydrogen atoms from instance
        """
        for cid, c in self:
            for rid, r in c:
                for aid, a in r:
                    if a.element == 'H':
                        print('removing H at %s' % aid)
                        r.remove_atom(a)

    def translate_xyz(self, xyz: XYZ) -> None:
        """
        :param xyz: a point
        :return: None. translates all pdb points by x, y and z
        """
        for cid, c in self:
            c.translate_xyz(xyz)

    def add_memb_res(self, memb_res: MembraneResidue) -> None:
        self.memb_res = memb_res

        # go over all residues, assign membrane Z value
        for cid in sorted(self.chains.keys()):
            for rid, res in sorted(self[cid].residues.items()):
                if -15. <= res['CA'].xyz.z <= 15:
                    res.memb_z = res['CA'].xyz.z
                else:
                    res.memb_z = None
        memb_chain = Chain(memb_res.chain, {'memb_res': memb_res})
        self.chains[memb_res.chain] = memb_chain

    def summarize(self):
        print('MyPDB instance with:')
        print('\t%i chains' % len(self.chains))
        print('\tsequences %s' % '\n\t'.join('>%s\n%s' % (k, v) for k, v in
                                             self.seqs.items()))
        print('\tmembrnae residue \n%r' % self.memb_res)

    def count_atoms_near_res(self, a_res: Residue, cutoff: float) -> int:
        atoms_set = set()
        for i_res in self.iter_all_res():

            if a_res.res_num-4 <= i_res.res_num <= a_res.res_num + 4:
                continue

            for aid, a_ in i_res:
                if a_res['CA'].distance(a_) <= cutoff:
                    atoms_set.add(aid)
        return len(atoms_set)

    def get_seq(self) -> dict:
        seqs = OrderedDict()
        for k, v in self.seqs.items():
            seqs[k] = v.get_seq()
        return seqs

    def get_AAseq(self) -> dict:
        return self.seqs

    def get_aaseqs(self) -> AASeqs:
        return self.aaseqs

def dihedral(p0: XYZ, p1: XYZ, p2: XYZ, p3: XYZ) -> float:
    """
    used http://www.cgl.ucsf.edu/Outreach/pc204/lecture_notes/phipsi/structured
    /phipsi.py
    :param p0: mp.XYZ
    :param p1: mp.XYZ
    :param p2: mp.XYZ
    :param p3: mp.XYZ
    :return: the dihedral angle between p0-3 in degrees
    >>> p0 =
    """
    from math import acos, degrees
    v01 = p0 - p1
    v32 = p3 - p2
    v12 = p1 - p2
    v0 = v12.cross(v01)
    v3 = v12.cross(v32)
    angle = degrees(acos(v0.dot(v3) / abs(v0) / abs(v3)))
    if v0.cross(v3).dot(v12) > 0:
        angle = -angle
    return angle


def distance_two_point_to_point(p1: XYZ, p2: XYZ, x: XYZ) -> float:
    """
    :param p1:point 1 (mp.XYZ) on line
    :param p2:point 2 (mp.XYZ) on line
    :param x:point x (mp.XYZ)
    :return: the minimal distance between the line pw-p1 and point x
    >>> p1 = mp.XYZ(0, 0, 0)
    >>> p2 = mp.XYZ(1, 0, 0)
    >>> x = mp.XYZ(1, 1, 0)
    >>> distance_two_point_to_point(p1, p2, x)
    1.0
    >>> x = mp.XYZ(-1, -1, 0)
    >>> distance_two_point_to_point(p1, p2, x)
    1.0
    >>> p2 = mp.XYZ(-1, 0, 0)
    >>> distance_two_point_to_point(p1, p2, x)
    1.0
    """
    Ul = (p2 - p1).unit()  # the unit vector between p2 to p1
    w = x - p1  # the vector from p1 to x
    return abs(Ul.cross(w))
