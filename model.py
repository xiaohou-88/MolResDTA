import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric

class SMILESModel(nn.Module):           
    def __init__(self, char_set_len):
        super().__init__()
       
class FASTAModel(nn.Module):
    def __init__(self, char_set_len, embed_size=128, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        
class Classifier(nn.Sequential):
    def __init__(self, smiles_model, fasta_model):
        super().__init__()
    
        
