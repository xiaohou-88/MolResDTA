from torch.utils.data import Dataset
import torch
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from torch_geometric.data import Data, Batch

def collate_fn(batch):
    """
    Custom collate function for handling graph data batches
    """
    # Unpack batch data
    smiles_fps, graphs, fastas, labels = zip(*batch)
    batched_smiles = torch.stack(smiles_fps)
    batched_graphs = Batch.from_data_list(graphs)
    batched_fastas = torch.stack(fastas)
    batched_labels = torch.stack(labels)
    return batched_smiles, batched_graphs, batched_fastas, batched_labels


__all__ = ['DTAData', 'collate_fn', 'CHARPROTSET', 'CHARPROTLEN', 'CHARISOSMISET', 'CHARISOSMILEN', 'MACCSLEN']

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
            "U": 19, "T": 20, "W": 21, 
            "V": 22, "Y": 23, "X": 24, 
            "Z": 25 }

CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

MACCSLEN = 2

def label_chars(chars, max_len, char_set):
    X = torch.zeros(max_len, dtype=torch.long)
    for i, ch in enumerate(chars[:max_len]):
        X[i] = char_set[ch]
    return X

def smiles_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor([int(_) for _ in fp.ToBitString()[1:]])

class DTAData(Dataset):             

    def __init__(self, smiles, fasta, label, device, max_smiles_len=100, max_fasta_len=1000):
        self.smiles = smiles
        self.fasta = fasta
        self.label = torch.tensor(label)
        self.device = device
        self.max_smiles_len = max_smiles_len
        self.max_fasta_len = max_fasta_len

    def __len__(self):
         return len(self.label)

    def __getitem__(self, idx):
        x, edge_index = smiles_to_graph(self.smiles[idx])
        graph_data = Data(
            x = x.to(self.device),
            edge_index = edge_index.to(self.device)
        )
        return (smiles_fingerprint(self.smiles[idx]).to(self.device),
                graph_data,
                label_chars(self.fasta[idx], self.max_fasta_len, CHARPROTSET).to(self.device),
                self.label[idx].float().to(self.device)
               )
    
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Get atom features
    num_atoms = mol.GetNumAtoms()
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic())
        ]
        atom_features.append(features)
    
    # Get bond connections
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
        
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    return x, edge_index
