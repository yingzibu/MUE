"""
Date: 09/07/2023
Mods: 02/21/2024: Add Morgan FP calculation
      
Athr: Bu
Aims: datasets and dataloaders for different model
Clas: 
      nn_dataset:  dataset for Multilayer Perceptron (MLP)
      GIN_dataset: dataset for Graph Isomorphism Network (GIN)
      rnn_dataset: dataset for Recurrent Neural  Network (RNN)
      
Func:
      get_loader:       get loader  for certain model type
      get_multi_loader: get loaders for train, valid, test
      smile_list_to_MACCS: calculate MACCS(FP) from SMILES
      procss: cal MACCSFP for data prepare for MLP dataset
      collate_molgraphs:   helper function for AttentiveFP
      get_AttentiveFP_dataset: get dataset for AttentiveFP
      get_AttentiveFP_loader:  get loader  for AttentiveFP
      get_GIN_dataloader:      get loader  for GIN
      get_rnn_loader:          get loader  for RNN
      
      ...

"""

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.MACCSkeys import GenMACCSKeys
import torch.nn.functional as F
import time
import dgl
import torch
import torch.nn as nn
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph, EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgllife.data import MoleculeCSVDataset
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from scripts.get_vocab import get_c2i_i2c, string2tensor

from dgllife.model import load_pretrained
from dgl.nn.pytorch.glob import AvgPooling
from functools import partial
from scripts.CONSTANT import *

def get_loader(df, names, params, model_type, vocab=None, vocab_type=None):
    print('--> preparing data loader for model type ', model_type)
    if model_type == 'MLP': return DataLoader(nn_dataset(df, names), **params)
    elif model_type == 'AttentiveFP':
        return get_AttentiveFP_loader(df, names, **params)

    elif model_type == 'GIN': 
        return get_GIN_dataloader(GIN_dataset(df, names), **params)

    elif model_type in ['RNN', 'VAE', 'RNN_pretrain']: 
        return get_rnn_loader(df, names, vocab, vocab_type, **params)
    
    else: print('invalid model type! '); return None

def get_multi_loader(trains, valids, tests, config):
    names = config['prop_names']
    vocab = None if 'vocab' not in config else config['vocab']
    v_t = None if 'vocab_type' not in config else config['vocab_type']
    batch_size = config['batch_size']
    model_type = config['model_type']

    print('---> loader for', names)
    params_ = {'batch_size': batch_size, 'shuffle': True,
               'drop_last': False, 'num_workers': 0}
    param_t = {'batch_size': batch_size, 'shuffle': False,
               'drop_last': False, 'num_workers': 0}

    # NEED TO CHANGE HERE TO INCLUDE SELFIES
    if model_type == 'RNN'and vocab == None:
        df = pd.concat([trains, valids, tests], ignore_index=True, axis=0)
        vocab = get_vocab(df, vocab_type=v_t)
    train_loader = get_loader(trains, names, params_, model_type, vocab, v_t)
    valid_loader = get_loader(valids, names, params_, model_type, vocab, v_t)
    test_loader  = get_loader(tests,  names, param_t, model_type, vocab, v_t)
    return train_loader, valid_loader, test_loader, vocab

"""functions and dataset for MLP"""
m = Chem.MolFromSmiles
header = ['bit' + str(i) for i in range(167)]
MASK = -100

def smile_list_to_MACCS(smi_list:list):
    MACCS_list = []
    for smi in smi_list:
        maccs = [float(i) for i in list(GenMACCSKeys(m(smi)).ToBitString())]
        MACCS_list.append(maccs)
    return MACCS_list

def process(data_):
    data = data_.copy()
    # data = convert_with_qed_sa(data)
    print('---> converting SMILES to MACCS...')
    MACCS_list = smile_list_to_MACCS(data['Drug'].tolist())
    data[header] = pd.DataFrame(MACCS_list)
    print('---> FINISHED')
    return data

def smile_list_to_MORGAN(smi_list, morgan_fp_len=MORGAN_LEN, radius=RADIUS):
    import rdkit
    from rdkit import Chem
    from tqdm import tqdm
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as MorganFP
    MORGAN_list = []
    for smi in tqdm(smi_list, total=len(smi_list)):
        mol = m(smi)
        morgan = [float(i) for i in list(MorganFP(m(smi), useChirality=True, 
                                        radius=radius, nBits=morgan_fp_len))]
        
        MORGAN_list.append(morgan)
    return MORGAN_list

def process_Morgan(data_, header=header_MORGAN):
    data = data_.copy()
    print('---> converting SMILES to Morgan FP...')
    l = smile_list_to_MORGAN(data['Drug'].tolist())
    len_here = len(l[0])
    assert len_here == len(header)
    data[header] = pd.DataFrame(l)
    print('---> FINISHED')
    
    return data

class nn_dataset(Dataset):
    def __init__(self, df, prop_names, mask=MASK, 
                 process=process, header=header):
        super(nn_dataset, self).__init__()
        df = process(df) # calculating MACCS
        df = df.fillna(mask)
        self.df = df
        self.len = len(df)
        self.fp = self.df[header]
        if isinstance(prop_names, str): prop_names = [prop_names]
        self.props = self.df[prop_names]

    def __getitem__(self, idx):
        fp = torch.tensor(self.fp.iloc[idx], dtype=torch.float32)
        label = torch.tensor(self.props.iloc[idx], dtype=torch.float32)
        return fp, label

    def __len__(self): return self.len

    def get_df(self): return self.df


"""Dataset and dataloader for Attentive FP"""
def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
        # masks = (labels == MASK).long()
    return smiles, bg, labels, masks

def get_AttentiveFP_dataset(df, name):
    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
    time_string = time.strftime("%m_%d_%Y_%H:%M:%S", time.localtime())

    params = {'smiles_to_graph': smiles_to_bigraph,
            'node_featurizer': atom_featurizer,
            'edge_featurizer': bond_featurizer,
            'smiles_column': 'Drug',
            'cache_file_path': time_string+'.bin',
            'task_names': name, 'load': True, 'n_jobs': len(name)*2}
    graph_dataset = MoleculeCSVDataset(df, **params)
    return graph_dataset

def get_AttentiveFP_loader(df, name, **loader_params):
    dataset = get_AttentiveFP_dataset(df, name)
    loader_params['collate_fn'] = collate_molgraphs
    loader = DataLoader(dataset, **loader_params)
    return loader


"""Dataset and dataloader for GIN pretrained model"""
class GIN_dataset(Dataset):
    def __init__(self, df, names, mask=MASK):
        df = df.fillna(mask)
        self.names = names
        self.df = df
        self.len = len(df)
        self.props = self.df[names]
        self.node_featurizer = PretrainAtomFeaturizer()
        self.edge_featurizer = PretrainBondFeaturizer()
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
    def __len__(self): return self.len
    
    def __getitem__(self, idx):
        v_d = self.df.iloc[idx]['Drug']
        v_d = self.fc(smiles=v_d, node_featurizer = self.node_featurizer,
                      edge_featurizer = self.edge_featurizer)
        label = torch.tensor(self.props.iloc[idx], dtype=torch.float32)
        return v_d, label

def get_GIN_dataloader(datasets, **loader_params):
    def dgl_collate_func(data):
        x, labels = map(list, zip(*data))
        bg = dgl.batch(x)
        labels = torch.stack(labels, dim=0)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        return bg, labels
    loader_params['collate_fn'] = dgl_collate_func
    return DataLoader(datasets, **loader_params)


"""Dataset and loader for RNN"""
class rnn_dataset(Dataset):
    def __init__(self, df, prop_names, vocab, vocab_type, 
                 device='cpu', mask=MASK):
        super(rnn_dataset, self).__init__()
        self.df = df.fillna(mask)
        self.device = device
        self.len = len(df)
        for col_smi in ['smiles', 'Drug', 'SMILES', 'Smiles', 'smile']: 
            if col_smi in self.df.columns: self.smi = self.df[col_smi]; break
        self.props = self.df[prop_names]
        self.c2i, _ =  get_c2i_i2c(vocab)
        self.vocab_type = vocab_type
    
    def __getitem__(self, idx):
        smi = self.smi[idx]
        tensor = string2tensor(smi, self.c2i, self.vocab_type, self.device)
        labels = torch.tensor(self.props.iloc[idx], 
                              dtype=torch.float32).to(self.device)
        return [tensor, labels]
    
    def __len__(self): return self.len

def get_rnn_loader(train, names, vocab, vocab_type, **loader_params):
    df = train.copy()
    dataset = rnn_dataset(df, names, vocab, vocab_type)
    c2i, _ =  get_c2i_i2c(vocab)
    def my_collate(batch):
        data = [item[0] for item in batch]
        data = pad_sequence(data, batch_first=True, 
                        padding_value=c2i['<pad>'])
        targets = [item[1] for item in batch]
        targets = torch.stack(targets)
        return (data, targets)
    
    loader_params['collate_fn'] = my_collate
    loader = DataLoader(dataset, **loader_params)
    return loader    

