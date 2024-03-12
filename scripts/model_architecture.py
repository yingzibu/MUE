import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# print(os.getcwd()) 
import time
import pandas as pd
import dgl
from torch.utils.data import DataLoader
from dgllife.model import model_zoo, load_pretrained
from dgllife.utils import smiles_to_bigraph, EarlyStopping, Meter
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.data import MoleculeCSVDataset
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from dgl.nn.pytorch.glob import AvgPooling
from torch.utils.data import Dataset, DataLoader
from functools import partial

from scripts.CONSTANT import MASK
# from scripts.train import get_loss_fn
from scripts.get_vocab import get_c2i_i2c
from scripts.dataset import get_rnn_loader

def get_loss_fn(IS_R):
    if IS_R: return nn.MSELoss(reduction='sum')
    else: return nn.BCEWithLogitsLoss(reduction='sum')

class Classifier(nn.Module):
    def __init__(self, **config):
        super(Classifier, self).__init__()
        dims = [config['in_dim'], config['hid_dims'], config['out_dim']]
        self.dims = dims
        neurons = [config['in_dim'], *config['hid_dims']]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) \
                         for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.final = nn.Linear(config['hid_dims'][-1], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        self.model_type = config['model_type']

    def forward(self, x):
        # for MUE: 
        ensemble_here = False
        if len(x.shape) == 3:
            ensemble_here = True
            batch_size, num_tasks, num_models = x.shape
            x = x.view(-1, num_models) 
        for layer in self.hidden: x = F.relu(layer(x))
        x = self.dropout(x)
        x = self.final(x)
        if ensemble_here == True: x = x.view(batch_size, num_tasks)
        return x

    def get_dim(self): return self.dims


def get_model_AT_10_17(names, n_layers, graph_feat_size, num_timesteps, dropout):
    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
    n_feats_num = atom_featurizer.feat_size('hv')
    e_feats_num = bond_featurizer.feat_size('he')

    model = model_zoo.AttentiveFPPredictor(
            node_feat_size=n_feats_num, edge_feat_size=e_feats_num,
            num_layers=n_layers, num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=len(names), dropout=dropout)
    
    return model

def AttentiveFP(**config):
    model =   get_model_AT_10_17(config['prop_names'], config['n_layers'],
    config['graph_feat_size'], config['num_timesteps'], config['dropout'])
    model.model_type = config['model_type']
    return model

# https://lifesci.dgl.ai/_modules/dgllife/model/pretrain.html
class GIN_MOD(nn.Module):
    """
    Reference: https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/encoders.py#L392
    """
	## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
    def __init__(self, **config):
        super(GIN_MOD, self).__init__()
        self.gnn = load_pretrained(config['pretrain_model'])
        self.readout = AvgPooling()
        self.transform = nn.Linear(300, config['in_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        self.hidden_dims = config['hid_dims']
        self.out_dim = config['out_dim']
        self.model_type = config['model_type']
        if len(self.hidden_dims) == 0:
            self.hidden = None
            self.final = nn.Linear(config['in_dim'], self.out_dim)
        else:
        # layer_size = len(self.hidden_dims)
            neurons = [config['in_dim'], *self.hidden_dims]
            linear_layers = [nn.Linear(neurons[i-1], neurons[i]) \
                                for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)
            self.final = nn.Linear(self.hidden_dims[-1], self.out_dim)

    def forward(self, bg, return_emb=False):
        # bg = bg.to(device)
        node_feats = [
            bg.ndata.pop('atomic_number'), bg.ndata.pop('chirality_type')]
        edge_feats = [
            bg.edata.pop('bond_type'), bg.edata.pop('bond_direction_type')]

        node_feats = self.gnn(bg, node_feats, edge_feats)
        x = self.readout(bg, node_feats)
        x = self.transform(x)
        if return_emb: return x
        if self.hidden != None:
            for layer in self.hidden: x = F.leaky_relu(layer(x))
        
        x = self.dropout(x)
        
        return self.final(x)

class RNN(nn.Module): 
    def __init__(self, **config): 
        super(RNN, self).__init__()
        self.vocab = config['vocab']
        n_vocab    = len(self.vocab)
        # vector     = torch.eye(n_vocab)
        self.bidir = config['Bidirect']
        self.device = config['device']
        self.GRU_dim = config['GRU_dim']
        self.num_layers = config['num_layers']
        self.model_type = config['model_type']
        self.vocab_type = config['vocab_type']
        self.c2i, self.i2c = get_c2i_i2c(self.vocab)
        self.x_emb = nn.Embedding(n_vocab, n_vocab, self.c2i['<pad>'])
        self.x_emb.weight.data.copy_(torch.eye(n_vocab).to(self.device))
        
        self.gru = nn.GRU(n_vocab, self.GRU_dim, num_layers=self.num_layers,
          batch_first=True, dropout=config['dropout'], bidirectional=self.bidir)
        self.hid_dim = self.GRU_dim * (2 if self.bidir else 1)
        if 'hid_dims' not in config: hid_dims = [self.GRU_dim]  
        else: hid_dims = config['hid_dims']
        neurons = [self.hid_dim, *hid_dims]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i])\
                        for i in range(1, len(neurons))]
        self.fc = nn.ModuleList(linear_layers)
        self.final = nn.Linear(neurons[-1], config['out_dim'])
    def forward(self, x):
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        _, x = self.gru(x, None)
        x = x[-(1 + int(self.gru.bidirectional)):]
        x = torch.cat(x.split(1), dim=-1).squeeze(0)
        for layer in self.fc: x = F.relu(layer(x))
        return self.final(x)

class RNNEncoder(nn.Module):
    def __init__(self, **config):
        super(RNNEncoder, self).__init__()
        self.vocab = config['vocab']
        n_vocab = len(self.vocab)
        self.bidir = config['Bidirect']
        self.device = config['device']
        self.GRU_dim = config['GRU_dim']

        self.num_layers = config['num_layers']
        self.vocab_type = config['vocab_type']
        self.c2i, self.i2c = get_c2i_i2c(self.vocab)

        self.x_emb = nn.Embedding(n_vocab, n_vocab, self.c2i['<pad>'])
        self.x_emb.weight.data.copy_(torch.eye(n_vocab).to(self.device))

        self.gru = nn.GRU(n_vocab, self.GRU_dim, num_layers=self.num_layers,
                    batch_first=True, bidirectional=self.bidir,
                    dropout=config['dropout'] if self.num_layers>1 else 0)
        self.hid_dim = self.GRU_dim * (2 if self.bidir else 1)
        if 'z_dim' not in config: self.z_dim = self.GRU_dim
        else: self.z_dim = config['z_dim']
        self.header = ['enc' + str(i) for i in range(self.z_dim)]
        self.mu = nn.Linear(self.hid_dim, self.z_dim)
        self.logvar = nn.Linear(self.hid_dim, self.z_dim)


    def forward(self, x, y=None):
        x = x.to(self.device)
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        _, x = self.gru(x, None)
        x = x[-(1 + int(self.gru.bidirectional)):]
        x = torch.cat(x.split(1), dim=-1).squeeze(0)
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar

    def cal_mu(self, df_:pd.DataFrame):
        df = df_.copy(); names = []; mu_list = []
        for col in df.columns:
            if col in ['smiles', 'selfies', 'Drug', 'SMILES', 'SELFIES']: pass
            else: names.append(col)
        ps = {'batch_size': 128, 'shuffle': False,
              'drop_last': False, 'num_workers': 0}
        loader = get_rnn_loader(df, names, self.vocab, self.vocab_type, **ps)
        
        for i, j in loader:
            i, j = i.to(self.device), j.to(self.device)
            mu, _ = self.forward(i)
            mu = mu.cpu().detach().numpy().tolist()
            mu_list += mu
        df[self.header] = pd.DataFrame(mu_list)
        return df

# config = get_config('VAE', name)

class RNNDecoder(nn.Module):
    def __init__(self, **config):
        super(RNNDecoder, self).__init__()
        self.vocab = config['vocab']
        self.vocab_type = config['vocab_type']
        n_vocab = len(self.vocab)
        self.c2i, self.i2c = get_c2i_i2c(self.vocab)
        self.pad_value = self.c2i['<pad>']
        self.device = config['device']
        self.x_emb = nn.Embedding(n_vocab, n_vocab, self.pad_value)
        self.x_emb.weight.data.copy_(torch.eye(n_vocab).to(self.device))

        if 'z_dim' not in config: self.z_dim = self.GRU_dim
        else: self.z_dim = config['z_dim']
        self.hid_dim = config['decoder_hid_dim']
        self.num_layers = config['decoder_num_layers']
        self.dropout = config['decoder_dropout']

        self.decoder_latent = nn.Linear(self.z_dim, self.hid_dim)

        self.gru = nn.GRU(n_vocab + self.z_dim, self.hid_dim, 
                          num_layers=self.num_layers, batch_first=True, 
                          dropout=self.dropout if self.num_layers>1 else 0)
        
        self.final = nn.Linear(self.hid_dim, n_vocab)

    def forward(self, x, z):
        x = x.to(self.device)
        z = z.to(self.device)
        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, 
                                      padding_value= self.pad_value)
        x_original = x.clone().detach().to(self.device)
        z_original = z.clone().detach().to(self.device)
        x = self.x_emb(x)
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, z], dim=-1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        z = self.decoder_latent(z_original)
        del z_original
        z = z.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        x, _ = self.gru(x, z)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.final(x)
        recon_loss = F.cross_entropy(x[:,:-1].contiguous().view(-1, x.size(-1)),
                                     x_original[:,1:].contiguous().view(-1),
                                     ignore_index = self.pad_value) 
        return x, recon_loss   

class RNNVAE(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.model_type = config['model_type']
        self.device = config['device']
        self.encoder = RNNEncoder(**config).to(self.device)
        self.decoder = RNNDecoder(**config).to(self.device)
        
        in_dim, out_dim =config['z_dim'], config['out_dim']
        self.IS_R = config['IS_R']
        if 'MLP_hid_dims' not in config: hid_dims = [128, 64, 32, 16]
        else: hid_dims = config['MLP_hid_dims']
        if 'MLP_dropout' not in config: MLP_dropout = 0
        else: MLP_dropout = config['MLP_dropout']
        classifier_config = {'model_type': 'MLP',
                             'in_dim': in_dim,
                             'out_dim': out_dim,
                             'hid_dims': hid_dims,
                             'dropout': MLP_dropout}
        
        self.classifier = Classifier(**classifier_config).to(self.device)
        
    def reparam(self, mu, logvar): # for encoder output
        eps = torch.randn_like(mu).to(self.device)
        z = mu + (logvar / 2).exp() * eps
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        del eps
        return z,  kl_loss
    
    def forward_encoder(self, x):
        mu, logvar = self.encoder(x)
        z, kl_loss = self.reparam(mu, logvar)
        del logvar
        return z, mu, kl_loss

    def forward(self, x, y=None):
        z, mu, kl_loss = self.forward_encoder(x)
        if y != None: y = self.classifier(mu) # calculate pred from classifier
            # classify_loss = self.classifier(mu, y, self.IS_R, return_pred=False)
        # else: classify_loss = torch.tensor([0]).to(self.device)
        _, recon_loss = self.decoder(x, z)

        return kl_loss, recon_loss, y

class RNN_pretrain(nn.Module):
    def __init__(self, **config): 
        super().__init__()
        self.model_type = config['model_type']
        self.device = config['device']
        self.encoder = RNNEncoder(**config).to(self.device)
        in_dim, out_dim =config['z_dim'], config['out_dim']
        self.IS_R = config['IS_R']
        if 'MLP_hid_dims' not in config: hid_dims = [128, 64, 32, 16]
        else: hid_dims = config['MLP_hid_dims']
        if 'MLP_dropout' not in config: MLP_dropout = 0
        else: MLP_dropout = config['MLP_dropout']
        classifier_config = {'model_type': 'MLP',
                             'in_dim': in_dim,
                             'out_dim': out_dim,
                             'hid_dims': hid_dims,
                             'dropout': MLP_dropout}
        
        self.classifier = Classifier(**classifier_config).to(self.device)

    # neccesary or not? 
    def load_model(self, encoder_path, classifier_path):
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=self.device))
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=self.device))
        print('finish loading: ', encoder_path, classifier_path)
    
    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.classifier(x)
        return x

