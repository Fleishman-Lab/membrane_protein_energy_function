"""
a class to describe an insertion profile of a single residue into the membrane
"""
from MPs.SplineCalibrationVars import *
import sys
import numpy as np
from scipy import interpolate
from utils.Logger import lgr


class InsertionProfile:
    """
    a single residue insertion profile described by either a sorted list of
    PoZEnergy instances (poz_energy) or by polynum
    """
    def __init__(self, aa: str, pos_score: dict,
                 membrane_half_depth: int=MEMBRANE_HALF_WIDTH,
                 residue_num: int=NUM_AAS, adjust_extra_membranal=True,
                 poly_edges: list=[]):
        """

        """
        self.AA = aa
        self.membrane_half_depth = membrane_half_depth
        self.residue_num = residue_num
        self.pos_score = pos_score
        self.extramembrane_adjusted = False
        self.poly_edges = poly_edges

    def __repr__(self):
        res = '<InsertionProfile for %s>\n' % self.AA
        res += '\t<poz_energy>\n'
        # for a in self.poz_energy:
        #     res += '\t\t<%s/>\n' % a
        res += '\t<poz_energy/>\n'
        # print('ddd', self.polynom)
        try:
            res += '<polynom: %.2f*z^4 + %.2f*z^3 %.2f*z^2 %.2f*z + %.2f />' % self.polynom
        except:
            pass
        res += '<InsertionProfile for %s/>\n' % self.AA
        return res

    def within_edges(self, pnt) -> bool:
        return self.poly_edges[0] <= pnt <= self.poly_edges[1]

    def polynom_at_z(self, z: float) -> float:
        """
        returns the polynoms value at z
        :type z: float
        """
        return np.polyval(self.polynom, z)

    def format_polyval(self):
        """
        print the polyval values for table
        """
        return ' '.join([str(a) for a in self.polynom])

    def format_spline_energies(self):
        """
        :return: string of all energies separated by spaces
        """
        if self.AA not in SKIP_AAS:
            return ' '.join(str(self.pos_score[pos]
                                if -SPLINE_LIM <= POS_Z_TOT[pos] <=
                                SPLINE_LIM else 0.0) for pos in POS_RANGE)
        else:
            lgr.log("creating %s spline as 0.0" % self.AA)
            return ' '.join("0.0" for pos in POS_RANGE)

    def rmsd_ips(self, other) -> float:
        """
        retruns the difference between the IPs calculated as by RMSD over Z
        """
        res = 0.0
        for pos in range(1, TOTAL_AAS+1):
            if self.poly_edges[0] <= POS_Z_TOT[pos] <= self.poly_edges[1]:
                res += (self.pos_score[pos] - other.pos_score[pos])**2
        return np.sqrt(np.mean(res))

    def adjust_exta_membrane(self):
        """
        set all positions outside [-20, 20] to 0. use only for setting splines in Rosetta
        :return:
        """
        self.extramembrane_adjusted = True
        print('ADJUSTING !!! !STOP ME !!!!')
        sys.exit()
        for pos in POS_RANGE:
            if -15 > POS_Z_TOT[pos] or +15 < POS_Z_TOT[pos]:
                self.pos_score[pos] = 0


# def pos_energy_dict_to_PoZEnergy_list(pos_energy_dict: dict) -> list():
#     """
#     creates an ordered list of PoZEnergy instances corresponding to their positions
#     """
#     result = []
#     for pos in range(1, TOTAL_AAS+1):
#         result.append(PoZEnergy(pos, POS_Z_TOT[pos], pos_energy_dict[pos]))
#     return result


def subtract_IP_from_IP(ip1: InsertionProfile, ip2: InsertionProfile, verbose: bool = False, smooth: bool=True) -> InsertionProfile:
    """
    """
    new_pos_score = {}
    if not smooth:
        for pos in POS_RANGE:
            new_pos_score[pos] = ip1.pos_score[pos] - ip2.pos_score[pos]
    else:
        # smooth the transition from water to membrane between +/-15A to
        # +/-25A for resulting splines
        y, x = [], []
        for pos in POS_RANGE:
            if -SPLINE_LIM > POS_Z_TOT[pos] or POS_Z_TOT[pos] > SPLINE_LIM:
                y.append(0.0)
                x.append(pos)
            elif ip1.poly_edges[0] <= POS_Z_TOT[pos] <= ip1.poly_edges[1]:
                y.append(ip1.pos_score[pos] - ip2.pos_score[pos])
                x.append(pos)
        tck = interpolate.splrep(x, y, s=SPLINE_SMOOTHNESS)
        new_pos_score = {pos: interpolate.splev(pos, tck)
                         if -SPLINE_LIM <= POS_Z_TOT[pos] <= +SPLINE_LIM else 0.0
                         for pos in POS_RANGE}

    return InsertionProfile(ip1.AA, new_pos_score)


def add_IP_to_IP(ip1: InsertionProfile, ip2: InsertionProfile, verbose: bool = False) -> InsertionProfile:
    """
    """
    new_pos_score = {}
    for pos in POS_RANGE:
        new_pos_score = ip1.pos_score[pos] = ip2.pos_score[pos]
        if verbose:
            print(pos, ip1.pos_score[pos], ip2.pos_score[pos],
                  ip1.pos_score[pos]+ip2.pos_score[pos])
    return InsertionProfile(ip1.AA, new_pos_score)

