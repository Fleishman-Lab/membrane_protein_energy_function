"""
a class for describing a membrane protein's topology
"""
import os
from glob import glob
from typing import List, Dict
import urllib3
import xml.etree.ElementTree
from PDB.MyPDB import MyPDB
from MPs.MPSpan import create_span, Orientation, Span


MY_HOME = os.path.expanduser('~')
PDBTM_LOCAL = '%s/elazaridis/mp_db/pdbtm' % MY_HOME


class MPTopo:
    def __init__(self, spans: List[Span], chain: str=None):
        self.spans = spans
        self.chain = chain

    def __str__(self) -> str:
        msg = 'MPTopo\n'
        for spn in self.spans:
            msg += '%s\n' % spn
        return msg

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, item) -> Span:
        return self.spans[item]

    def starts(self) -> List[int]:
        """starts
        a list of span starts
        :rtype: Listint
        """
        return [spn.start for spn in self.spans]

    def ends(self) -> List[int]:
        """ends
        a list of span ends
        :rtype: Listint
        """
        return [spn.end for spn in self.spans]

    def orientations(self) -> List[str]:
        """orientations
        a list of span orientations
        :rtype: Liststr
        """
        return [spn.orientation for spn in self.spans]

    def get_script_vars(self, chain=None) -> str:
        """get_script_vars
        get a string for script vars description of all spans
        :param chain: chain name to use

        :rtype: str
        """
        if not chain and self.chain:
            chain = self.chain
        else:
            chain = ''
        starts = ','.join(['%i%s' % (s, chain) for s in self.starts()])
        ends = ','.join(['%i%s' % (s, chain) for s in self.ends()])
        msg = '-script_vars span_starts=%s\n' % starts
        msg += '-script_vars span_ends=%s\n' % ends
        msg += '-script_vars span_oris=%s\n' % ','.join(map(str,
                                                            self.orientations()))
        return msg

    def get_pymol_select(self, pdb_name: str=None, chain: str=None) -> str:
        """get_pymol_select
        get a string of a pymol selection for all spans
        :param pdb_name: pdb name
        :type pdb_name: str
        :param chain: chain name
        :type chain: str

        :rtype: str
        """
        if not chain and self.chain:
            chain = self.chain
        else:
            chain = ''
        spans_str = '+'.join(['%i-%i' % (span.start, span.end)
                              for span in self.spans])
        return 'select spans, %s///%s/' % (pdb_name, spans_str)


def topo_from_pdbtm(pdb: MyPDB, name: str=None) -> Dict[str, MPTopo]:
    """topo_from_pdbtm

    :param pdb: pdb instance
    :type pdb: MyPDB
    :param name: name
    :type name: str

    :rtype: DictstrMPTopo
    >>> from PDB.MyPDB_funcs import parse_PDB
    >>> pdb = parse_PDB('/home/labs/fleishman/jonathaw/scripts/tests/2vt4.pdb')
    >>> topo = topo_from_pdbtm(pdb, '2vt4')  # doctest: +IGNORE_RESULT
    >>> topo['A'][0].start
    41
    >>> topo['A'][0].end
    63
    required to ignore print output in test
    """.replace('+IGNORE_RESULT', '+ELLIPSIS\n<...>')
    chain_spans = find_spans_by_pdbtm_xml(pdb, name)
    chain_mptopo = {}
    for chain in pdb.chains.keys():
        new_spans = get_chain_spans_from_pdb_spans(chain, chain_spans, pdb)
        chain_mptopo[chain] = MPTopo(new_spans, chain)
    return chain_mptopo


def find_spans_by_pdbtm_xml(pdb: MyPDB,
                            name: str=None,
                            xml_file: str=None) -> Dict[str, List]:
    """find_spans_by_pdbtm_xml
    parses a PDBTM xml to create a dictionary of chain: List[Span]

    :param pdb: pdb instance
    :type pdb: MyPDB
    :param name: pdb code
    :type name: str
    :param xml_file: xml file
    :type xml_file: str

    :rtype: DictstrList
    """
    # find xml file or url
    if not xml_file:
        xml_file = glob('%s/database/*/%s.xml' % (PDBTM_LOCAL, pdb.name))
        if not xml_file:
            xml_file = glob('%s/database/*/%s.xml' % (PDBTM_LOCAL, name))
        xml_tree = xml.etree.ElementTree.parse(xml_file[0]).getroot()
    else:
        xml_tree = xml.etree.ElementTree.parse(xml_file).getroot()
    if not xml_file:
        html = 'http://pdbtm.enzim.hu/data/database/%s/%s.xml' % (name[1:-1],
                                                                  name)
        http = urllib3.PoolManager()
        xml_html = http.request('GET', html)
        xml_tree = xml.etree.ElementTree.fromstring(
            xml_html.data.decode('utf-8'))

    # read side tag to decide if in this xml the inside is side 1 or 2 and vice
    # versa
    side_dict = {'1': None, '2': None}
    for side in xml_tree.iter('{http://pdbtm.enzim.hu}SIDEDEFINITION'):
        side1 = side.get('Side1')
        side2 = side.get('Side2')
    if side1 == 'Outside' or side2 == 'Inside':
        side_dict['1'] = 'out'
        side_dict['2'] = 'in'
    elif side1 == 'Inside' or side2 == 'Outside':
        side_dict['1'] = 'in'
        side_dict['2'] = 'out'
    else:
        print('xml side unclear for xml_file')

    # parse xml for spans
    topo = []
    all_chains = {}
    for ch in xml_tree.iter('{http://pdbtm.enzim.hu}CHAIN'):
        chain = ch.get('CHAINID')
        regions = ch.findall('{http://pdbtm.enzim.hu}REGION')
        span_lst = []
        for reg in regions:
            topo.append(reg.get('type'))
            if reg.get('type') == 'H':
                span_lst.append({'start': int(reg.get('pdb_beg')),
                                 'end': int(reg.get('pdb_end'))})
            else:
                span_lst.append(None)
        # go over spans from the xml, make them into Span instances
        topo = fix_unkowns(topo)
        spans = []
        for i, spn in enumerate(span_lst):
            if spn:
                ori = '%s2%s' % (side_dict[topo[i-1]], side_dict[topo[i+1]])
                if ori == 'in2out':
                    ori_ = Orientation(1)
                elif ori == 'out2in':
                    ori_ = Orientation(2)
                else:
                    ori_ = Orientation(3)

                spans.append(create_span(pdb, spn['start'], spn['end'], ori_,
                                         span_num=i+1, chain=chain))
        all_chains[chain] = spans
    return all_chains


def get_chain_spans_from_pdb_spans(chain: str,
                                   chain_spans: Dict[str, List[Span]],
                                   pdb: MyPDB) -> List[Span]:
    """get_chain_spans_from_pdb_spans
    in case the required chain is not mentioned in the PDBTM xml,
    a more complex approach isrequired:
    1. find the chain with span entries in the xml which is most similar to the
       query chain
    2. using the xml data for the chosen chain, find the corresponding
       positions in the query pdb
    3. return as a list of Span objects

    :param chain: the required chain
    :type chain: str
    :param chain_spans: a dictionary with chain: List[Span] entries from xml
    :type chain_spans: Dict[str, List[Span]]
    :param pdb: the pdb as a MyPDB instance
    :type pdb: mp.MyPDB

    :rtype: List[Span]
    """
    if chain in chain_spans.keys():
        return chain_spans[chain]
    print('chain not found in xml, looking for similar chains')
    full_aaseqs = pdb.aaseqs
    query_seq = full_aaseqs[chain]
    best_score, best_chain = 0, ''
    for chain_w_span in chain_spans.keys():
        temp_seq = full_aaseqs[chain_w_span]
        print(temp_seq, full_aaseqs.keys())
        score, _, _ = query_seq.align(temp_seq)
        if score > best_score:
            best_chain = chain_w_span
            best_seq = temp_seq
            best_score = score
    print('found the best match %s with score %i' % (best_chain, best_score))

    updated_spans = []

    # for every span:
    # 1. find the start (and end) position in the pdb sequence from the xml pdb
    # position
    # 2. find the corresponding in the sequence alignment with the query chain
    # 3. find the pdb position for the query chain
    for span in chain_spans[best_chain]:
        start_pdb_num_a = span.start

        start_seq_num_a = pdb[best_chain].seq_pos_from_pdb_pos(start_pdb_num_a)

        start_seq_num_b = query_seq.aligned_position_at_non_aligned(
            best_seq.non_aligned_position_at_aligned(start_seq_num_a))

        start_pdb_num_b = list(pdb[chain].residues.keys())[start_seq_num_b]

        end_pdb_num_a = span.end

        end_seq_num_a = pdb[best_chain].seq_pos_from_pdb_pos(end_pdb_num_a)

        end_seq_num_b = query_seq.aligned_position_at_non_aligned(
            best_seq.non_aligned_position_at_aligned(end_seq_num_a))

        end_pdb_num_b = list(pdb[chain].residues.keys())[end_seq_num_b]

        temp_span = create_span(pdb, start_pdb_num_b, end_pdb_num_b,
                                span.orientation, span_num=span.span_num,
                                chain=chain)
        updated_spans.append(temp_span)
    return updated_spans


def fix_unkowns(topo_: List[str]) -> List[str]:
    new_topo = topo_[:]
    for i, a in enumerate(topo_):
        if a == 'U':
            for ind in range(i, 0, -1):
                if topo_[ind] in '12':
                    new_topo[i] = topo_[ind]
            for ind in range(i, len(topo_)):
                if topo_[ind] in '12':
                    new_topo[i] = topo_[ind]
    return new_topo
