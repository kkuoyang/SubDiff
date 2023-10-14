import faulthandler
faulthandler.enable()
import multiprocessing as mp
import os
import os.path as path
import pickle
from collections import Counter
from datetime import datetime
from functools import partial
from typing import List, Tuple

from tqdm import tqdm
import numpy as np
from arguments import parse_arguments
# from model.mydataclass import Paths

# from model.mol_graph import MolGraph

"""For molecular graph processing."""
import faulthandler
faulthandler.enable()
import os
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import rdkit.Chem as Chem
import torch
from rdkit.Chem import Descriptors
from tqdm import tqdm

from merging_operation_learning import merge_nodes
# from model.mydataclass import *
# from model.utils import (fragment2smiles, get_conn_list, graph2smiles,
                        #  networkx2data, smiles2mol)
# from model.vocab import MotifVocab, SubMotifVocab, Vocab

RDContribDir = os.path.join(os.environ['CONDA_PREFIX'], 'share', 'RDKit', 'Contrib')
sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
# import sascorer
from rdkit.Contrib.SA_Score import sascorer
from dataclasses import dataclass

from torch import LongTensor, Tensor
sys.path.pop()
from typing import Dict, List, Tuple

import networkx as nx
import rdkit.Chem as Chem
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit
import rdkit.Chem as Chem
from typing import List, Tuple, Dict
import torch
# from model.utils import smiles2mol, get_conn_list
from collections import defaultdict

import torch_geometric

from torch_geometric.data import Data
from torch_geometric.data import Batch

class Vocab(object):
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vmap = dict(zip(self.vocab_list, range(len(self.vocab_list))))
        
    def __getitem__(self, smiles):
        # print('==========')
        # print(smiles)
        # print(self.vmap)
        # print(self.vmap[smiles])
        # print('==========')
        return self.vmap.get(smiles)

    def get_smiles(self, idx):
        return self.vocab_list[idx]

    def size(self):
        return len(self.vocab_list)

class MotifVocab(object):

    def __init__(self, pair_list: List[Tuple[str, str]]):
        self.motif_smiles_list = [motif for _, motif in pair_list]
        self.motif_vmap = dict(zip(self.motif_smiles_list, range(len(self.motif_smiles_list))))

        node_offset, conn_offset, num_atoms_dict, nodes_idx = 0, 0, {}, []
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}
        conn_dict: Dict[int, Tuple[int, int]] = {}
        bond_type_motifs_dict = defaultdict(list)
        for motif_idx, motif_smiles in enumerate(self.motif_smiles_list):
            motif = smiles2mol(motif_smiles)
            ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))

            cur_orders = []
            vocab_conn_dict[motif_idx] = {}
            for atom in motif.GetAtoms():
                if atom.GetSymbol() == '*' and ranks[atom.GetIdx()] not in cur_orders:
                    bond_type = atom.GetBonds()[0].GetBondType()
                    vocab_conn_dict[motif_idx][ranks[atom.GetIdx()]] = conn_offset
                    conn_dict[conn_offset] = (motif_idx, ranks[atom.GetIdx()])
                    cur_orders.append(ranks[atom.GetIdx()])
                    bond_type_motifs_dict[bond_type].append(conn_offset)
                    nodes_idx.append(node_offset)
                    conn_offset += 1
                node_offset += 1
            num_atoms_dict[motif_idx] = motif.GetNumAtoms()
        self.vocab_conn_dict = vocab_conn_dict
        self.conn_dict = conn_dict
        self.nodes_idx = nodes_idx
        self.num_atoms_dict = num_atoms_dict
        self.bond_type_conns_dict = bond_type_motifs_dict


    def __getitem__(self, smiles: str) -> int:
        if smiles not in self.motif_vmap:
            print(f"{smiles} is <UNK>")
        return self.motif_vmap[smiles] if smiles in self.motif_vmap else -1
    
    def get_conn_label(self, motif_idx: int, order_idx: int) -> int:
        return self.vocab_conn_dict[motif_idx][order_idx]
    
    def get_conns_idx(self) -> List[int]:
        return self.nodes_idx
    
    def from_conn_idx(self, conn_idx: int) -> Tuple[int, int]:
        return self.conn_dict[conn_idx]

class SubMotifVocab(object):

    def __init__(self, motif_vocab: MotifVocab, sublist: List[int]):
        self.motif_vocab = motif_vocab
        self.sublist = sublist
        self.idx2sublist_map = dict(zip(sublist, range(len(sublist))))

        node_offset, conn_offset, nodes_idx = 0, 0, []
        motif_idx_in_sublist = {}
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}
        for i, mid in enumerate(sublist):
            motif_idx_in_sublist[mid] = i
            vocab_conn_dict[mid] = {}
            for cid in motif_vocab.vocab_conn_dict[mid].keys():
                vocab_conn_dict[mid][cid] = conn_offset
                nodes_idx.append(node_offset + cid)
                conn_offset += 1
            node_offset += motif_vocab.num_atoms_dict[mid]
        self.vocab_conn_dict = vocab_conn_dict
        self.nodes_idx = nodes_idx
        self.motif_idx_in_sublist_map = motif_idx_in_sublist
    
    def motif_idx_in_sublist(self, motif_idx: int):
        return self.motif_idx_in_sublist_map[motif_idx]

    def get_conn_label(self, motif_idx: int, order_idx: int):
        return self.vocab_conn_dict[motif_idx][order_idx]
    
    def get_conns_idx(self):
        return self.nodes_idx
def smiles2mol(smiles: str, sanitize: bool=False) -> Chem.rdchem.Mol:
    if sanitize:
        return Chem.MolFromSmiles(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    AllChem.SanitizeMol(mol, sanitizeOps=0)
    return mol

def graph2smiles(fragment_graph: nx.Graph, with_idx: bool=False) -> str:
    motif = Chem.RWMol()
    node2idx = {}
    for node in fragment_graph.nodes:
        idx = motif.AddAtom(smarts2atom(fragment_graph.nodes[node]['smarts']))
        if with_idx and fragment_graph.nodes[node]['smarts'] == '*':
            motif.GetAtomWithIdx(idx).SetIsotope(node)
        node2idx[node] = idx
    for node1, node2 in fragment_graph.edges:
        motif.AddBond(node2idx[node1], node2idx[node2], fragment_graph[node1][node2]['bondtype'])
    return Chem.MolToSmiles(motif, allBondsExplicit=True)

def networkx2data(G: nx.Graph) -> Tuple[Data, Dict[int, int]]:
    num_nodes = G.number_of_nodes()
    mapping = dict(zip(G.nodes(), range(num_nodes)))
    
    G = nx.relabel_nodes(G, mapping)
    G = G.to_directed() if not nx.is_directed(G) else G

    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    x = torch.tensor([i for _, i in G.nodes(data='label')])
    edge_attr = torch.tensor([[i] for _, _, i in G.edges(data='label')], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, mapping

def fragment2smiles(mol: Chem.rdchem.Mol, indices: List[int]) -> str:
    smiles = Chem.MolFragmentToSmiles(mol, tuple(indices))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))

def smarts2atom(smarts: str) -> Chem.rdchem.Atom:
    return Chem.MolFromSmarts(smarts).GetAtomWithIdx(0)

def mol_graph2smiles(graph: nx.Graph, postprocessing: bool=True) -> str:
    mol = Chem.RWMol()
    graph = nx.convert_node_labels_to_integers(graph)
    node2idx = {}
    for node in graph.nodes:
        idx = mol.AddAtom(smarts2atom(graph.nodes[node]['smarts']))
        node2idx[node] = idx
    for node1, node2 in graph.edges:
        mol.AddBond(node2idx[node1], node2idx[node2], graph[node1][node2]['bondtype'])
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)
    return postprocess(smiles) if postprocessing else smiles
 
def postprocess(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and not atom.IsInRing():
                atom.SetIsAromatic(False)   
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                if not (bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()):
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        
        for _ in range(100):
            problems = Chem.DetectChemistryProblems(mol)
            flag = False
            for problem in problems:
                if problem.GetType() =='KekulizeException':
                    flag = True
                    for atom_idx in problem.GetAtomIndices():
                        mol.GetAtomWithIdx(atom_idx).SetIsAromatic(False)
                    for bond in mol.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                            if not (bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()):
                                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol), sanitize=False)
            if flag: continue
            else: break
        
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
        except:
            print(f"{smiles} not valid")
            return "CC"
        smi = Chem.MolToSmiles(mol)
        return smi

def get_conn_list(motif: Chem.rdchem.Mol, use_Isotope: bool=False, symm: bool=False) -> Tuple[List[int], Dict[int, int]]:

    ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))
    if use_Isotope:
        ordermap = {atom.GetIsotope(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    else:
        ordermap = {atom.GetIdx(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    if len(ordermap) == 0:
        return [], {}
    ordermap = dict(sorted(ordermap.items(), key=lambda x: x[1]))
    if not symm:
        conn_atoms = list(ordermap.keys())
    else:
        cur_order, conn_atoms = -1, []
        for idx, order in ordermap.items():
            if order != cur_order:
                cur_order = order
                conn_atoms.append(idx)
    return conn_atoms, ordermap


def label_attachment(smiles: str) -> str:

    mol = Chem.MolFromSmiles(smiles)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    dummy_atoms = [(atom.GetIdx(), ranks[atom.GetIdx()])for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    dummy_atoms.sort(key=lambda x: x[1])
    orders = []
    for (idx, order) in dummy_atoms:
        if order not in orders:
            orders.append(order)
            mol.GetAtomWithIdx(idx).SetIsotope(len(orders))
    return Chem.MolToSmiles(mol)

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    _, preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()

    number, indices = torch.topk(scores, k=10, dim=-1)
    topk_acc = torch.eq(indices, labels.view(-1,1)).float()
    return torch.sum(acc) / labels.nelement(), torch.sum(topk_acc) / labels.nelement()

def sample_from_distribution(distribution: torch.Tensor, greedy: bool=False, topk: int=0):
    if greedy or topk == 1:
        motif_indices = torch.argmax(distribution, dim=-1)
    elif topk == 0 or len(torch.where(distribution > 0)) <= topk:
        motif_indices = torch.multinomial(distribution, 1)
    else:
        _, topk_idx = torch.topk(distribution, topk, dim=-1)
        mask = torch.zeros_like(distribution)
        ones = torch.ones_like(distribution)
        mask.scatter_(-1, topk_idx, ones)
        motif_indices = torch.multinomial(distribution * mask, 1)
    return motif_indices

@dataclass
class train_data:
    graph: Data
    query_atom: int
    cyclize_cand: List[int]
    label: Tuple[int, int]

@dataclass
class mol_train_data:
    mol_graph: Data
    props: Tensor
    start_label: int
    train_data_list: List[train_data]
    motif_list: List[int]

ATOM_SYMBOL_VOCAB = Vocab(['*', 'N', 'O', 'Se', 'Cl', 'S', 'C', 'I', 'B', 'Br', 'P', 'Si', 'F'])
ATOM_ISAROMATIC_VOCAB = Vocab([True, False])
ATOM_FORMALCHARGE_VOCAB = Vocab(["*", -1, 0, 1, 2, 3])
ATOM_NUMEXPLICITHS_VOCAB = Vocab(["*", 0, 1, 2, 3])
ATOM_NUMIMPLICITHS_VOCAB = Vocab(["*", 0, 1, 2, 3])
ATOM_FEATURES = [ATOM_SYMBOL_VOCAB, ATOM_ISAROMATIC_VOCAB, ATOM_FORMALCHARGE_VOCAB, ATOM_NUMEXPLICITHS_VOCAB, ATOM_NUMIMPLICITHS_VOCAB]
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_VOCAB = Vocab(BOND_LIST)

LOGP_MEAN, LOGP_VAR = 3.481587226600002, 1.8185146774225027
MOLWT_MEAN, MOLWT_VAR = 396.7136355500001, 110.55283206754517
QED_MEAN, QED_VAR = 0.5533041888502863, 0.21397359224960685
SA_MEAN, SA_VAR = 2.8882909807901354, 0.8059540682960904

class MolGraph(object):

    @classmethod
    def load_operations(cls, operation_path: str, num_operations: int=500):
        MolGraph.NUM_OPERATIONS = num_operations
        MolGraph.OPERATIONS = [code.strip('\r\n') for code in open(operation_path)]
        MolGraph.OPERATIONS = MolGraph.OPERATIONS[:num_operations]
    
    @classmethod
    def load_vocab(cls, vocab_path: str):
        pair_list = [line.strip("\r\n").split() for line in open(vocab_path)]
        MolGraph.MOTIF_VOCAB = MotifVocab(pair_list)
        MolGraph.MOTIF_LIST = MolGraph.MOTIF_VOCAB.motif_smiles_list

    def __init__(self,
        smiles: str,
        tokenizer: str="graph",
    ):  
        assert tokenizer in ["graph", "motif"], \
            "The variable `process_level` should be 'graph' or 'motif'. "
        self.smiles = smiles
        self.mol = smiles2mol(smiles, sanitize=True)
        self.mol_graph = self.get_mol_graph()
        self.init_mol_graph = self.mol_graph.copy()
        
        if tokenizer == "motif":
            self.merging_graph = self.get_merging_graph()
            self.refragment()
            self.motifs = self.get_motifs()

    def get_mol_graph(self) -> nx.Graph:
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        for atom in self.mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['smarts'] = atom.GetSmarts()
            graph.nodes[atom.GetIdx()]['atom_indices'] = set([atom.GetIdx()])
            graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(atom)

        for bond in self.mol.GetBonds():
            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            graph[atom1][atom2]['bondtype'] = bond.GetBondType()
            graph[atom1][atom2]['label'] = BOND_VOCAB[bond.GetBondType()]

        return graph
    
    def get_merging_graph(self) -> nx.Graph:
        mol = self.mol
        mol_graph = self.mol_graph.copy()
        merging_graph = mol_graph.copy()
        for code in self.OPERATIONS:
            for (node1, node2) in mol_graph.edges:
                if not merging_graph.has_edge(node1, node2):
                    continue
                atom_indices = merging_graph.nodes[node1]['atom_indices'].union(merging_graph.nodes[node2]['atom_indices'])
                pattern = Chem.MolFragmentToSmiles(mol, tuple(atom_indices))
                if pattern == code:
                    merge_nodes(merging_graph, node1, node2)
            mol_graph = merging_graph.copy()
        return nx.convert_node_labels_to_integers(merging_graph)

    def refragment(self) -> None:
        mol_graph = self.mol_graph.copy()
        merging_graph = self.merging_graph

        for node in merging_graph.nodes:
            atom_indices = self.merging_graph.nodes[node]['atom_indices']
            merging_graph.nodes[node]['motif_no_conn'] = fragment2smiles(self.mol, atom_indices)
            for atom_idx in atom_indices:
                mol_graph.nodes[atom_idx]['bpe_node'] = node

        for node1, node2 in self.mol_graph.edges:
            bpe_node1, bpe_node2 = mol_graph.nodes[node1]['bpe_node'], mol_graph.nodes[node2]['bpe_node']
            if bpe_node1 != bpe_node2:
                conn1 = len(mol_graph)
                mol_graph.add_node(conn1)
                mol_graph.add_edge(node1, conn1)

                conn2 = len(mol_graph)
                mol_graph.add_node(conn2)
                mol_graph.add_edge(node2, conn2)
                
                mol_graph.nodes[conn1]['smarts'] = '*'
                mol_graph.nodes[conn1]['targ_atom'] = node2
                mol_graph.nodes[conn1]['merge_targ'] = conn2
                mol_graph.nodes[conn1]['anchor'] = node1
                mol_graph.nodes[conn1]['bpe_node'] = bpe_node1
                mol_graph[node1][conn1]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                mol_graph[node1][conn1]['label'] = mol_graph[node1][node2]['label']
                merging_graph.nodes[bpe_node1]['atom_indices'].add(conn1)
                mol_graph.nodes[conn1]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)
                
                mol_graph.nodes[conn2]['smarts'] = '*'
                mol_graph.nodes[conn2]['targ_atom'] = node1
                mol_graph.nodes[conn2]['merge_targ'] = conn1
                mol_graph.nodes[conn2]['anchor'] = node2
                mol_graph.nodes[conn2]['bpe_node'] = bpe_node2
                mol_graph[node2][conn2]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                mol_graph[node2][conn2]['label'] = mol_graph[node1][node2]['label']
                merging_graph.nodes[bpe_node2]['atom_indices'].add(conn2)
                mol_graph.nodes[conn2]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)

        for node in merging_graph.nodes:
            atom_indices = merging_graph.nodes[node]['atom_indices']
            motif_graph = mol_graph.subgraph(atom_indices)
            merging_graph.nodes[node]['motif'] = graph2smiles(motif_graph)

        self.mol_graph = mol_graph

    def get_motifs(self) -> Set[str]:
        return [(self.merging_graph.nodes[node]['motif_no_conn'], self.merging_graph.nodes[node]['motif']) for node in self.merging_graph.nodes]

    def relabel(self):
        mol_graph = self.mol_graph
        bpe_graph = self.merging_graph

        for node in bpe_graph.nodes:
            bpe_graph.nodes[node]['internal_edges'] = []
            atom_indices = bpe_graph.nodes[node]['atom_indices']
            
            fragment_graph = mol_graph.subgraph(atom_indices)
            motif_smiles_with_idx = graph2smiles(fragment_graph, with_idx=True)
            motif_with_idx = smiles2mol(motif_smiles_with_idx)
            conn_list, ordermap = get_conn_list(motif_with_idx, use_Isotope=True)
           
            bpe_graph.nodes[node]['conn_list'] = conn_list
            bpe_graph.nodes[node]['ordermap'] = ordermap
            bpe_graph.nodes[node]['label'] = MolGraph.MOTIF_VOCAB[ bpe_graph.nodes[node]['motif'] ]
            bpe_graph.nodes[node]['num_atoms'] = len(atom_indices)

        for node1, node2 in bpe_graph.edges:
            self.merging_graph[node1][node2]['label'] = 0

        edge_dict = {}
        for edge, (node1, node2, attr) in enumerate(mol_graph.edges(data=True)):
            edge_dict[(node1, node2)] = edge_dict[(node2, node1)] = edge
            bpe_node1 = mol_graph.nodes[node1]['bpe_node']
            bpe_node2 = mol_graph.nodes[node2]['bpe_node']
            if bpe_node1 == bpe_node2:
                bpe_graph.nodes[bpe_node1]['internal_edges'].append(edge)
        
        for node, attr in mol_graph.nodes(data=True): 
            if attr['smarts'] == '*':
                anchor = attr['anchor']
                targ_atom = attr['targ_atom']
                mol_graph.nodes[node]['edge_to_anchor'] = edge_dict[(node, anchor)]
                mol_graph.nodes[node]['merge_edge'] = edge_dict[(anchor, targ_atom)]
    
    def get_props(self) -> List[float]:
        mol = self.mol
        logP = (Descriptors.MolLogP(mol) - LOGP_MEAN) / LOGP_VAR
        Wt = (Descriptors.MolWt(mol) - MOLWT_MEAN) / MOLWT_VAR
        qed = (Descriptors.qed(mol) - QED_MEAN) / QED_VAR
        sa = (sascorer.calculateScore(mol) - SA_MEAN) / SA_VAR
        properties = [logP, Wt, qed, sa]
        return properties

    def get_data(self) -> mol_train_data:
        
        self.relabel()
        init_mol_graph, mol_graph, bpe_graph = self.init_mol_graph, self.mol_graph, self.merging_graph
        init_mol_graph_data, _ = networkx2data(init_mol_graph)
        motifs_list, conn_list = [], []
        train_data_list: List[train_data] = []

        nodes_num_atoms = dict(bpe_graph.nodes(data='num_atoms'))
        node = max(nodes_num_atoms, key=nodes_num_atoms.__getitem__)
        start_label = bpe_graph.nodes[node]['label']
        motifs_list.append(start_label)

        conn_list.extend(self.merging_graph.nodes[node]['conn_list'])
        subgraph = nx.Graph()
        subgraph = nx.union(subgraph, mol_graph.subgraph(bpe_graph.nodes[node]['atom_indices']))

        while len(conn_list) > 0:
            query_atom = conn_list[0]
            targ = mol_graph.nodes[query_atom]['merge_targ']
            
            subgraph_data, mapping = networkx2data(subgraph)

            if targ in conn_list:  
                cur_mol_smiles_with_idx = graph2smiles(subgraph, with_idx=True)
                motif_with_idx = smiles2mol(cur_mol_smiles_with_idx)
                _, ordermap = get_conn_list(motif_with_idx, use_Isotope=True)

                cyc_cand = [mapping[targ]]
                for cand in conn_list[1:]:
                    if ordermap[cand] != ordermap[targ]:
                        cyc_cand.append(mapping[cand])
                
                train_data_list.append(train_data(
                    graph = subgraph_data,
                    query_atom = mapping[query_atom],
                    cyclize_cand = cyc_cand,
                    label = (-1, 0),
                ))
     
            else:
                node = mol_graph.nodes[targ]['bpe_node']
                motif_idx = bpe_graph.nodes[node]['label']
                motifs_list.append(motif_idx)
                ordermap = bpe_graph.nodes[node]['ordermap']
                conn_idx = ordermap[targ]
                cyc_cand = [mapping[cand] for cand in conn_list[1:]]

                train_data_list.append(train_data(
                    graph = subgraph_data,
                    query_atom = mapping[query_atom],
                    cyclize_cand = cyc_cand,
                    label = (motif_idx, conn_idx),
                ))

                conn_list.extend(bpe_graph.nodes[node]['conn_list'])
                subgraph = nx.union(subgraph, mol_graph.subgraph(bpe_graph.nodes[node]['atom_indices']))

            anchor1 = mol_graph.nodes[query_atom]['anchor']
            anchor2 = mol_graph.nodes[targ]['anchor']
            subgraph.add_edge(anchor1, anchor2)
            subgraph[anchor1][anchor2]['bondtype'] = mol_graph[anchor1][anchor2]['bondtype']
            subgraph[anchor1][anchor2]['label'] = mol_graph[anchor1][anchor2]['label']
            subgraph.remove_node(query_atom)
            subgraph.remove_node(targ)
            conn_list.remove(query_atom)
            conn_list.remove(targ)

        props = self.get_props()
        motifs_list = list(set(motifs_list))
        return mol_train_data(
            mol_graph = init_mol_graph_data,
            props = props,
            start_label = start_label,
            train_data_list = train_data_list,
            motif_list = motifs_list,
        )        

    @staticmethod
    def preprocess_vocab() -> Batch:
        vocab_data = []
        for idx in tqdm(range(len(MolGraph.MOTIF_LIST))):
            graph, _, _ = MolGraph.motif_to_graph(MolGraph.MOTIF_LIST[idx])
            data, _ = networkx2data(graph)
            vocab_data.append(data)
        vocab_data = Batch.from_data_list(vocab_data)
        return vocab_data

    @staticmethod
    def get_atom_features(atom: Chem.rdchem.Atom=None, IsConn: bool=False, BondType: Chem.rdchem.BondType=None) -> Tuple[int, int, int, int, int]:
        if IsConn:
            Symbol, FormalCharge, NumExplicitHs, NumImplicitHs = 0, 0, 0, 0       
            IsAromatic = True if BondType == Chem.rdchem.BondType.AROMATIC else False
            IsAromatic = ATOM_ISAROMATIC_VOCAB[IsAromatic]
        else:
            Symbol = ATOM_SYMBOL_VOCAB[atom.GetSymbol()]
            IsAromatic = ATOM_ISAROMATIC_VOCAB[atom.GetIsAromatic()]
            FormalCharge = ATOM_FORMALCHARGE_VOCAB[atom.GetFormalCharge()]
            NumExplicitHs = ATOM_NUMEXPLICITHS_VOCAB[atom.GetNumExplicitHs()]
            NumImplicitHs = ATOM_NUMIMPLICITHS_VOCAB[atom.GetNumImplicitHs()]
        return (Symbol, IsAromatic, FormalCharge, NumExplicitHs, NumImplicitHs)

    @staticmethod
    def motif_to_graph(smiles: str, motif_list: Optional[List[str]] = None) -> Tuple[nx.Graph, List[int], List[int]]:
        motif = smiles2mol(smiles)
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(motif))
        
        dummy_list = []
        for atom in motif.GetAtoms():
            idx = atom.GetIdx()
            graph.nodes[idx]['smarts'] = atom.GetSmarts()
            graph.nodes[idx]['motif'] = smiles
            if atom.GetSymbol() == '*':
                graph.nodes[idx]['dummy_bond_type'] = bondtype = atom.GetBonds()[0].GetBondType()
                graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)
                dummy_list.append(idx)
            else:
                graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(atom)
        
        ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False))
        dummy_list = list(zip(dummy_list, [ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*']))
        if len(dummy_list) > 0:
            dummy_list.sort(key=lambda x: x[1])
            dummy_list, _ = zip(*dummy_list)       

        for bond in motif.GetBonds():
            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            graph[atom1][atom2]['bondtype'] = bond.GetBondType()
            graph[atom1][atom2]['label'] = BOND_VOCAB[bond.GetBondType()]

        return graph, list(dummy_list), ranks


def apply_operations(batch: List[Tuple[int, str]], mols_pkl_dir: str) -> Counter:
    vocab = Counter()
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for idx, smi in batch:
            mol = MolGraph(smi, tokenizer="motif")
            with open(path.join(mols_pkl_dir, f"{idx}.pkl"), "wb") as f:
                pickle.dump(mol, f)
            vocab = vocab + Counter(mol.motifs)
            pbar.update()
    return vocab

def motif_vocab_construction(
    train_path: str,
    vocab_path: str,
    operation_path: str,
    num_operations: int,
    num_workers: int,
    mols_pkl_dir: str,
):

    print(f"[{datetime.now()}] Construcing motif vocabulary from {train_path}.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.")
    data_set = []
    files = ["train.smiles.npz", "valid.smiles.npz", "test.smiles.npz"]
    i = -1
    for path_file in files:
        cur_path = path.join(train_path, path_file)
        with np.load(cur_path) as f:
            for _, val in f.items():
                for smile in val:
                    i += 1
                    data_set.append((i, smile.strip("\n")))
    # data_set = [(idx, smi.strip("\n")) for idx, smi in enumerate(open(train_path))]
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    print(f"Total: {len(data_set)} molecules.\n")

    print(f"Processing...")
    vocab = Counter()
    os.makedirs(mols_pkl_dir, exist_ok=True)
    MolGraph.load_operations(operation_path, num_operations)
    func = partial(apply_operations, mols_pkl_dir=mols_pkl_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        for batch_vocab in pool.imap(func, batches):
            vocab = vocab + batch_vocab

    atom_list = [x for (x, _) in vocab.keys() if x not in MolGraph.OPERATIONS]
    atom_list.sort()
    new_vocab = []
    full_list = atom_list + MolGraph.OPERATIONS
    for (x, y), value in vocab.items():
        assert x in full_list
        new_vocab.append((x, y, value))
        
    index_dict = dict(zip(full_list, range(len(full_list))))
    sorted_vocab = sorted(new_vocab, key=lambda x: index_dict[x[0]])
    with open(vocab_path, "w") as f:
        for (x, y, _) in sorted_vocab:
            f.write(f"{x} {y}\n")
    
    print(f"\r[{datetime.now()}] Motif vocabulary construction finished.")
    print(f"The motif vocabulary is in {vocab_path}.\n\n")

if __name__ == "__main__":

    args = parse_arguments()
    # paths = Paths(args)
    preprocess_dir = "preprocess/"
    os.makedirs(preprocess_dir, exist_ok=True)

    motif_vocab_construction(
        train_path = "qm9/temp/qm9/",
        vocab_path = path.join(preprocess_dir, "vocab.txt"),
        operation_path = path.join(preprocess_dir, "merging_operation.txt"),
        num_operations = args.num_operations,
        mols_pkl_dir = path.join(preprocess_dir, "mol_graphs"),
        num_workers = args.num_workers,
    )

    
    
    