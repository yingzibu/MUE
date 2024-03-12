"""
Date: 09/07/2023
Mods: 11/08/2023, edited train_epoch_VAE: add tqdm
Athr: Bu
Aims: train functions for all models
Mods: 
      02/05/2024, add scale_back in model_predict function
Func:
      tbc...
"""

import yaml
import time
import torch
import IPython
import imageio
import numpy as np
import pandas as pd
from os import walk
import torch.nn as nn
from tqdm import tqdm
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dgllife.utils import EarlyStopping, Meter
from torch.utils.data import DataLoader, Dataset


from scripts.CONSTANT import *
from scripts.eval_utils import *
from scripts.func_utils import *
from scripts.dataset import get_loader
from scripts.model_architecture import *

clip_grad  = 50
model_types = ['MLP', 'AttentiveFP', 'GIN', 'RNN', 
                'VAE', 'RNN_pretrain', 'MUE']

def count_bool(lst): return sum(lst)

def count_parameters(model: Module):
    return sum(p.numel() for p in model.parameters())

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optim_params(model):
    return (p for p in model.parameters() if p.requires_grad)
    
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def clamp(number, min_value=1e-5, max_value=1-1e-5):
    return max(min_value, min(number, max_value))
# uncertainty weight
class MTLoss(Module): # calculate multitask loss with trainable parameters
    def __init__(self, task_num, weight_loss=None, device='cuda'): 
        super(MTLoss, self).__init__()
        self.task_num = task_num
        self.device = device
        if weight_loss==None: weight_loss = [1.0] * self.task_num
        # 𝜂 := 𝑙𝑜𝑔𝜎2
        self.eta = nn.Parameter(torch.tensor(weight_loss, device=self.device))
        
    def forward(self, loss_list, IS_R=None):
        assert len(loss_list) == self.task_num
        if IS_R != None: 
            assert len(IS_R) == len(loss_list)
            for i in range(len(IS_R)):
                if IS_R[i] == True: loss_list[i] /= 2 
        loss_list = torch.Tensor(loss_list).to(self.device)
        total_loss = loss_list * torch.exp(-self.eta) + self.eta * 0.5 

        weight_loss = [float(self.eta[i].item()) for i in range(self.task_num)]
        weight_loss = [np.exp(-1.0 * i) for i in weight_loss]
        if IS_R != None:
            for i in range(len(IS_R)):
                if IS_R[i] == True: weight_loss[i] /= 2

        weight_loss = [i/sum(weight_loss) for i in weight_loss]
        
        for n in range(len(weight_loss)): 
            if weight_loss[n] < 1e-5: weight_loss[n] = clamp(weight_loss[n])
        weight_loss = [i/sum(weight_loss) for i in weight_loss]

        
        return total_loss.sum(), weight_loss


def init_model(**config):
    """need incorporate all models here! """
    if   config['model_type'] == 'MLP':          model = Classifier(**config)
    elif config['model_type'] == 'GIN':          model = GIN_MOD(**config) 
    elif config['model_type'] == 'AttentiveFP':  model = AttentiveFP(**config)
    elif config['model_type'] == 'RNN':          model = RNN(**config)
    elif config['model_type'] == 'VAE':          model = RNNVAE(**config)
    elif config['model_type'] == 'RNN_pretrain': model = RNN_pretrain(**config)
    elif config['model_type'] == 'MUE':          model = Classifier(**config)
    else:                    print('invalid model type:', config['model_type'])
    return model


def get_train_fn(model_type):
    if   model_type == 'VAE':       return train_epoch_VAE
    elif model_type in model_types: return train_epoch_MLP
    else: print('invalid model type:', model_type); return

def get_eval_fn(model_type):
    if   model_type == 'VAE':       return train_epoch_VAE
    elif model_type in model_types: return train_epoch_MLP
    else: print('invalid model type:', model_type); return

def train_epoch_MLP(model, loader, IS_R, names, device,
                    epoch=None, optimizer=None, MASK=-100,
                    scale_dict=None, weight_loss=None, 
                    model_type='model', ver=False):
    """
    param weight_loss: list, the weight of loss for different tasks
    """
    
    if optimizer==None: # no optimizer, either validation or test
        model.eval()    # model evaluation for either valid or test
        if epoch != None: train_type='Valid' # if epoch is inputted, its valid
        else: train_type = 'Test' # if no epoch information, its test
    else: model.train(); train_type='Train' # if optimizer inputted, its train
    

    if isinstance(IS_R, list): IS_R_list = IS_R
    else: IS_R_list = [IS_R] * len(names)
    if weight_loss == None: weight_loss = [1.0/len(names)]*len(names)

    total_loss, losses_list, y_probs, y_label = 0, [], {}, {}
    for idx, batch_data in enumerate(loader):
        """
        len(batch_data) could determine which algorithm
        len(batch_data) == 2: MLP, GIN, RNN, ENSEMBLE
        len(batch_data) == 4: AttentiveFP
        """
        if len(batch_data) == 2:  # MLP or GIN or RNN
            fp, labels = batch_data
            fp, labels = fp.to(device), labels.to(device)
            mask = labels == MASK
            pred = model(fp)
        elif len(batch_data) == 4: # attentiveFP
            smiles, bg, labels, masks = batch_data
            bg,labels,masks = bg.to(device), labels.to(device), masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            pred = model(bg, n_feats, e_feats)
            mask = masks < 1
        
        batch_loss_list = []
        for j, (name, IS_R, w) in enumerate(zip(names, IS_R_list, weight_loss)):
            loss_func = get_loss_fn(IS_R)
            probs = pred[:, j][~mask[:, j]]
            label = labels[:, j][~mask[:, j]]
            
            len_here = label.shape[0] # num of data with labels
            loss_here = loss_func(probs, label) 
            if len_here != 0:
                loss_here /= len_here
                batch_loss_list.append(loss_here.item())
            else:       batch_loss_list.append(float(0))
            
            if j == 0: loss  = loss_here * w
            else:      loss += loss_here * w

            if IS_R == False: probs = F.sigmoid(probs)
            if train_type != 'Train': # valid or test, output probs and labels
                                      # if train, no process prob to save time
                probs = probs.cpu().detach().numpy().tolist()
                label = label.cpu().detach().numpy().tolist()
                if scale_dict != None:
                    if name in scale_dict.keys():
                        min_here = scale_dict[name][0]
                        max_here = scale_dict[name][1]
                        del_here = max_here - min_here
                        label = [l * del_here + min_here for l in label]
                        probs = [p * del_here + min_here for p in probs]
                    
                if idx == 0: y_probs[name], y_label[name] = probs, label
                else:     y_probs[name] += probs; y_label[name] += label
        
        if len(losses_list) == 0:               losses_list = batch_loss_list
        else: losses_list = [i+j for i, j in zip(losses_list, batch_loss_list)]

        total_loss += loss.item()

        if optimizer != None:
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    total_loss /= len(loader) # no need /loader.dataset since has / len_here 
    
    if epoch != None: # train or valid
        if ver: print(f'Epoch:{epoch}, [{train_type}] Loss: {total_loss:.3f}')
    
    elif epoch == None: # test
        print(f'[{train_type}] Loss: {total_loss:.3f}')
        performance = eval_dict(y_probs, y_label, names, IS_R_list, 
                                model_type=model_type, draw_fig=True)
        performance['loss'] = float(total_loss)

    IS_R = IS_R_list
    if   train_type == 'Train': return total_loss, losses_list, IS_R # train
    elif train_type == 'Valid': return total_loss,  y_probs, y_label # valid
    else:                       return performance, y_probs, y_label # test




def train_epoch_VAE(model, loader, IS_R, names, device, 
                    kl_weight=1, cls_weight=1,
                    epoch=None, optimizer=None, 
                    MASK=-100, scale_dict=None, 
                    weight_loss=None, model_type='model'):

    if optimizer == None: 
        model.eval()          # Valid or Test
        if epoch != None: train_type = 'Valid'
        else:             train_type = 'Test'
    else:  model.train(); train_type = 'Train'

    if isinstance(IS_R, list): IS_R_list = IS_R
    else: IS_R_list = [IS_R] * len(names)
    
    if weight_loss == None: weight_loss = [1.0/len(names)]*len(names)
    total_loss, losses_list, y_probs, y_label = 0, [], {}, {}
    klds, recs, clss = 0, 0, 0
    for idx, batch_data in tqdm(enumerate(loader), 
                            total=len(loader), desc=f'{train_type}'):
        if len(batch_data) == 2: x, labels = batch_data
        else: x, labels = batch_data, None
        kld, rec, pred = model(x, labels)
        if pred == None: cls = torch.tensor([0]).to(device)
        else: # label is not none, and has pred, calculate: 
            labels = labels.to(device)
            mask = labels == MASK
            batch_loss_list = []
            for j, (name, IS_R, w) in enumerate(
                zip(names, IS_R_list, weight_loss)):
                loss_func = get_loss_fn(IS_R)
                probs = pred[:,j][~mask[:,j]]
                label = labels[:,j][~mask[:,j]]
                len_here = label.shape[0]
                loss_here = loss_func(probs, label)
                if len_here != 0: 
                    loss_here /= len_here
                    batch_loss_list.append(loss_here.item())
                else: batch_loss_list.append(float(0))

                if j == 0: cls  = loss_here * w
                else:      cls += loss_here * w

                if IS_R == False: probs = F.sigmoid(probs)
                if train_type != 'Train': 
                    probs = probs.cpu().detach().numpy().tolist()
                    label = label.cpu().detach().numpy().tolist()
                    if scale_dict != None: 
                        if name in scale_dict.keys():
                            min_here = scale_dict[name][0]
                            max_here = scale_dict[name][1]
                            del_here = max_here - min_here
                            label = [l * del_here + min_here for l in label]
                            probs = [p * del_here + min_here for p in probs]
                    if idx == 0: y_probs[name], y_label[name] = probs, label
                    else:     y_probs[name] += probs; y_label[name] += label 
            if len(losses_list) == 0: losses_list = batch_loss_list
            else: losses_list = [i+j for i, j in zip(
                                    losses_list, batch_loss_list)]
            
        loss = kl_weight * kld + rec + cls_weight * cls
        total_loss += loss.item()
        klds += kld.item(); recs += rec.item(); clss += cls.item()

        if optimizer != None: 
            optimizer.zero_grad(); loss.backward()
            clip_grad_norm_(get_optim_params(model), clip_grad)
            optimizer.step()
        lr=(optimizer.param_groups[0]['lr'] if optimizer is not None else None)
    
    IS_R = IS_R_list
    total_loss /= len(loader)
    klds /= len(loader); recs /= len(loader); clss /= len(loader)
    if epoch != None: 
        print(f'Epoch:{epoch} [{train_type}] Loss: {total_loss:.3f} |',
              f'KL Div: {klds:.3f} | Recon: {recs:.3f} | Classify: {clss:.3f}',
              f'| KL w: {kl_weight:.3f} | cls w: {cls_weight:.3f}') 
    else:
        print(f'[{train_type}] Loss: {total_loss:.3f} | Classify: {clss:.3f}')
        perf = eval_dict(y_probs, y_label, names, IS_R, 
                         model_type=model_type, draw_fig=True)
        perf['loss'] = float(clss)
    
    if   train_type == 'Train': return [total_loss, clss], losses_list, IS_R
    elif train_type == 'Valid': return [total_loss, clss], y_probs, y_label
    elif train_type == 'Test':  return perf,       y_probs, y_label

def eval_VAE(model, loader, names, scale_dict=None, MASK=MASK):
    model.eval()
    if isinstance(names, str): names = [names]
    IS_R = [names_dict[name] for name in names]
    y_probs, y_label, mu_dict = {}, {}, {}
    for idx, (input_, labels) in enumerate(loader):
        input_, labels = input_.to(model.device), labels.to(model.device)
        mu, _ = model.encoder(input_)
        preds = model.classifier(mu)
        mask = labels == MASK
        del input_

        for j, (name, is_r) in enumerate(zip(names, IS_R)):
            probs = preds[:,j][~mask[:,j]]
            label = labels[:,j][~mask[:,j]]
            mask_here = mask[:,j].reshape(mask[:, j].shape[0], 1).expand_as(mu)
            mu_ = mu * (~mask_here)
            del mask_here
            if is_r == False: probs = F.sigmoid(probs)
            probs = probs.cpu().detach().numpy().tolist()
            label = label.cpu().detach().numpy().tolist()
            mu_   = mu_.cpu().detach().numpy()

            if scale_dict != None:
                if name in scale_dict.keys():
                    min_here = scale_dict[name][0]
                    max_here = scale_dict[name][1]
                    del_here = max_here - min_here
                    label = [l*del_here + min_here for l in label]
                    probs = [p*del_here + min_here for p in probs]
            if idx == 0:
                y_probs[name], y_label[name], mu_dict[name] = probs, label, mu_
            else:
                y_probs[name] += probs; y_label[name] += label
                mu_dict[name] = np.append(mu_dict[name], mu_, axis=0)

    return mu_dict, y_probs, y_label 

def model_predict(model, loader, IS_R, names, device, model_type, 
        MASK=-100, scale_dict=None, scale_back=True):
    if isinstance(IS_R, list): IS_R_list = IS_R
    else: IS_R_list = [IS_R] * len(names)
    model.eval()
    y_probs = {}
    for idx, batch_data in tqdm(enumerate(loader), total=len(loader), 
                                desc='Predicting...'):
        if len(batch_data) == 1: print('dataloader error'); return
        elif len(batch_data) == 2: 
            fp, labels = batch_data;    labels = labels.to(device)
            if model_type == 'VAE': _, _, pred = model(fp, labels)
            else: fp = fp.to(device);     pred = model(fp)
        elif len(batch_data) == 4: 
            _,bg,_,_=batch_data; bg = bg.to(device)
            # bg, labels = bg.to(device), labels.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            pred = model(bg, n_feats, e_feats)
        # print(pred.shape)
        # print(fp.shape)
        for j, (name, IS_R) in enumerate(zip(names, IS_R_list)):
            probs = pred[:, j]
            if IS_R == False: probs = F.sigmoid(probs)
            probs = probs.cpu().detach().numpy().tolist()
            if scale_dict != None and scale_back == True: 
                if name in scale_dict.keys():
                    min_here = scale_dict[name][0]
                    max_here = scale_dict[name][1]
                    del_here = max_here - min_here
                    probs = [p * del_here + min_here for p in probs]
            if idx == 0: y_probs[name]  = probs
            else:        y_probs[name] += probs
    return y_probs


class PRED:
    def __init__(self, **config):
        if 'device' in config: self.device = config['device']
        else: 
            cuda = torch.cuda.is_available()
            if cuda: self.device = 'cuda'
            else:    self.device = 'cpu'
        
        self.config = config; self.prop_names = config['prop_names']
        
        if 'scale_dict' not in config: self.scale_dict = None
        else:           self.scale_dict = config['scale_dict']
        if 'weight_loss' not in config: self.weight_loss = None
        else:          self.weight_loss = config['weight_loss']
        self.model_type = config['model_type']
        self.model = init_model(**config).to(self.device)
        self.params_num = count_parameters(self.model)
        print('Model type: ', self.model_type, end="")
        print(' | Model parameters: ',self.params_num)
        self.vocab = None if 'vocab' not in config else config['vocab']
        
        if 'vocab_type' not in config: self.vocab_type = None
        else:           self.vocab_type = config['vocab_type']
        
        # initialize model paths and config path
        self.model_path = config['model_path']
        if 'encoder_path' not in config: self.encoder_path = None
        else:          self.encoder_path = config['encoder_path']
        if 'decoder_path' not in config: self.decoder_path = None
        else:          self.decoder_path = config['decoder_path']
        if 'classifier_path' not in config: self.classifier_path = None
        else:          self.classifier_path = config['classifier_path']
        if 'config_path' not in config: 
            c_p = self.model_path.split('.')[0] + '.yml'
            self.config_path = c_p; self.config['config_path'] = c_p
        else: self.config_path = config['config_path']
        if 'figure_path' not in config:
            figure_path = self.model_path.split('.')[0]
            self.figure_path = figure_path
            self.config['figure_path'] = figure_path
        else: self.figure_path = config['figure_path']
        
        self.eval_fn = get_eval_fn(self.model_type)
        self.train_fn = get_train_fn(self.model_type)
        self.IS_R = config['IS_R'] # could be list, could be true/false
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                        lr=config['lr'], weight_decay=config['wd'])
        self.stopper = EarlyStopping(mode='lower', patience=config['patience'])
        if 'verbose_freq' not in config:
            self.verbose_freq = 10
            self.config['verbose_freq'] = self.verbose_freq
        else: self.verbose_freq = config['verbose_freq']
        if 'uncertainty_weight' not in config: 
            if len(self.prop_names) == 1: self.uw = False # single task
            else: self.uw = True
        else: self.uw = config['uncertainty_weight']    
        self.min_loss, self.best_epoch = np.inf, 0
        if 'MAX_EPOCH' not in self.config: self.MAX_EPOCH = 1000
        else:          self.MAX_EPOCH = self.config['MAX_EPOCH']
        
        if 'max_kl_weight' not in self.config: self.max_kl_weight = 0.5
        else: self.max_kl_weight = self.config['max_kl_weight']
        if 'cls_weight' not in self.config: self.cls_weight = 0.5
        else: self.cls_weight = self.config['cls_weight'] 

        self.train_dict, self.valid_dict, self.times_list = {}, {}, []
        
        # will store the results on test set, if test set == None, leave blank
        self.performance_dict = {} 

        self.data = dict(config     = self.config,
                         min_loss   = self.min_loss,
                         best_epoch = self.best_epoch, 
                         train_dict = self.train_dict, 
                         valid_dict = self.valid_dict,
                         times_list = self.times_list,
                         params_num = self.params_num, 
                         performance= self.performance_dict)
    
    def save_train_status(self): 
        self.data = dict(
            config = self.config,
            min_loss = self.min_loss,
            best_epoch = self.best_epoch, 
            train_dict = self.train_dict, 
            valid_dict = self.valid_dict,
            times_list = self.times_list,
            params_num = self.params_num, 
            performance = self.performance_dict)

        with open(self.config_path, 'w') as fl:
            yaml.dump(self.data, fl, default_flow_style=False)
        print('\n--> Train status saved at', self.config_path)

    def load_encoder(self, encoder_path=None):# this is VAE or RNN_pretrain
        if encoder_path == None: encoder_path = self.encoder_path 
        self.model.encoder.load_state_dict(torch.load(
               encoder_path, map_location=self.device))
        print('Finish load encoder from', encoder_path)
    
    def load_decoder(self, decoder_path=None):
        if decoder_path == None: decoder_path = self.decoder_path
        self.model.decoder.load_state_dict(torch.load(
               decoder_path, map_location=self.device))
        print('Finish load decoder from', decoder_path)

    def load_classifier(self, classifier_path=None):
        if classifier_path == None: classifier_path=self.classifier_path
        self.model.classifier.load_state_dict(torch.load(classifier_path, 
                                               map_location=self.device)) 
        print('Finish load classifier from', classifier_path)

    def load_vae_pretrain(self, encoder_path=None, classifier_path=None):
        self.load_encoder(encoder_path)
        self.load_classifier(classifier_path)
    
    def load_vae(self, encoder_path=None, decoder_path=None):
        self.load_encoder(encoder_path); self.load_decoder(decoder_path)

    def load_model(self, path=None):
        if self.model_type == 'AttentiveFP': 
            con = self.config.copy(); con['dropout'] = 0
            self.model = init_model(**con).to(self.device)
        if path == None: path = self.model_path
        print('load pretrained model from ', path)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def load_status(self, data):
        # with open(yml_file_path, 'r') as f:
        #     data = yaml.safe_load(f)
        self.data = data
        self.config = data['config']
        self.model_path = self.config['model_path'] 
        # self.load_model(self.config['model_path'])
        self.min_loss = data['min_loss']
        self.best_epoch = data['best_epoch']
        self.train_dict = data['train_dict']
        self.valid_dict = data['valid_dict']
        self.times_list = data['times_list']
        self.params_num = data['params_num']
        self.performance_dict = data['performance']
        print('finish load data status \n')

    def get_runtime(self, verbose=True):
        if verbose:
            print(f'Train time: {np.mean(self.times_list):.3f}'
                  f'+/-{np.std(self.times_list):.3f} ms')
        return np.mean(self.times_list), np.std(self.times_list)

    def print_config(self): 
        print('#'*68); print('#'*30, 'CONFIG', '#'*30); print('#'*68)
        for i, j in self.config.items():             print(i, ':', j)
        print('#'*68)

    def eval(self, loader, path=None, ver=False):
        if ver: self.print_config()
        
        if path != None: self.load_model(path)
        else: self.load_model(self.model_path)
        if self.model_type == 'VAE': self.load_vae_pretrain()
        if ver:
            # if self.weight_loss != None: print('task weight: ',self.weight_loss)
            print('Model parameters: ', count_parameters(self.model))
            self.get_runtime()
            print(f"epoch {self.best_epoch} -> min loss {self.min_loss:.4f}")
            plot_loss(self.train_dict, self.valid_dict, name='valid',
                      title_name= f'loss during training {self.model_type}')
        
        performance, probs, label = self.eval_fn(self.model, loader, self.IS_R, 
            self.prop_names, self.device, epoch=None, optimizer=None, 
            MASK=-100, scale_dict=self.scale_dict, weight_loss=None,
            # weight_loss=self.weight_loss, # should not use the weightloss
            model_type=self.model_type)
        return performance, probs, label


    def train_VAE(self, loader, val_loader, test_loader=None, 
                 data_df=pd.DataFrame(), sample_num=1, latent_eval_freq=1):
        """
        Aim: Train Variational AutoEncoder (VAE), evaluate latent space 
        params: 
            loader:       DataLoader, for train
            val_loader:   DataLoader, for valid, save model based on its loss
            test_loader:  DataLoader, for test,  final performance evaluation
            data_df: pd.DataFrame, if no empty, will plot PCA on latent sapce
            sample_num:       int, the number of entries sampled from data_df
            latent_eval_freq: int, the frequency of plot PCA and save figures
        Return performance: dict, performance on test loader
                            performance[name]: metrics nums evaluated on name
                            performance[loss]: total loss of VAE for test set 
        """
        
        if len(self.prop_names) == 1: self.uw = False 
        if self.uw: 
            m_w = MTLoss(len(self.prop_names), device=self.device) 
            optimizer = torch.optim.SGD(m_w.parameters(), lr=0.1); m_w.train()

        # self.optimizer = torch.optim.AdamW(
        #     get_optim_params(self.model), lr=lr_start)
        # lr_annealer = CosineAnnealingLRWithRestart(self.optimizer)
        
        self.model.zero_grad(); start_epoch = self.best_epoch
        min_total_loss = np.inf; train_total, valid_total = {}, {} 
        for epoch in range(start_epoch, self.MAX_EPOCH):
            
            if epoch % latent_eval_freq == 0:
                if len(data_df) > 0: # plot PCA before training one epoch 
                    if epoch % self.verbose_freq == 0: plot_show = True
                    else: plot_show = False
                    tmp_ = data_df.sample(n=sample_num).reset_index(drop=True)
                    tmp_ = self.model.encoder.cal_mu(tmp_)
                    header_here = self.model.encoder.header
                    for n in self.prop_names: 
                        # there might be MASK values in tmp, delete entire row
                        tmp = tmp_[tmp_[n]!= MASK]
                        plot_dim_reduced(tmp[header_here],tmp[n],names_dict[n],
                        dim_reduct='PCA',title = f'PCA on {n} in latent space',
                        savepath=self.figure_path, 
                        savename=f'PCA_{n}_{epoch}.png', plot_show=plot_show)

            inc = (epoch - start_epoch) / (self.MAX_EPOCH - start_epoch)
            kl_weight = self.max_kl_weight * inc
            # cls_weight = self.cls_weight
            cls_weight = self.cls_weight * inc
            t = time.time()
            score, l, r = self.train_fn(self.model, loader, self.IS_R,
                                        self.prop_names, self.device, kl_weight, 
                                        cls_weight, epoch, self.optimizer, 
                                        scale_dict  = self.scale_dict,
                                        weight_loss = self.weight_loss,
                                        model_type  = self.model_type)
            train_time = (time.time() - t) * 1000 / len(loader.dataset)
            
            val_s, probs, label = self.train_fn(
                self.model, val_loader, self.IS_R, self.prop_names, self.device, 
                kl_weight, cls_weight, epoch, scale_dict=self.scale_dict,
                weight_loss=self.weight_loss, model_type=self.model_type)
            
            
            self.times_list.append(train_time)
            if self.uw:
                optimizer.zero_grad()
                total_loss, self.weight_loss = m_w(l, r)
                total_loss.backward(); optimizer.step()
            

            self.train_dict[epoch] = score[1]; train_total[epoch] = score[0]
            self.valid_dict[epoch] = val_s[1]; valid_total[epoch] = val_s[0]
            if val_s[0] < min_total_loss:
                print(f'# SAVE MODEL: loss: {min_total_loss:.3f} ->',
                      f'{val_s[0]:.3f}', end=" | ")
                min_total_loss = val_s[0];  self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_path)

            if val_s[1] < self.min_loss: 
                # use cls loss as save indicator of enc, dec cls model
                # seperate save to simplify later load pretrain models
                # load model for MLP, just need to load encoder for FP
                # load model for RNN_pretrain, no need to load decoder
                # load model for VAE on different tasks, no classifier
                print(f'## SAVE Enc, Dec, Cls: classify loss:',
                      f'{self.min_loss:.3f} -> {val_s[1]:.3f} ',
                      f'| runtime: {train_time:.3f} ms')
                self.min_loss = val_s[1]
                torch.save(self.model.encoder.state_dict(), self.encoder_path)
                torch.save(self.model.decoder.state_dict(), self.decoder_path)
                torch.save(self.model.classifier.state_dict(), 
                           self.classifier_path)
            
            

            early_stop = self.stopper.step(val_s[1], self.model) # cls loss
            
            
            if epoch % self.verbose_freq == 0 and epoch != 0:
                self.get_runtime()
                if self.uw:                 print('different task weight', 
                            ['{:.3f}'.format(i) for i in self.weight_loss])
                plot_loss(self.train_dict, self.valid_dict, name='valid',
                title_name= f'Classify loss during training {self.model_type}')
                
                plot_loss(train_total, valid_total, name='valid',
                title_name= f'Total loss during training {self.model_type}')
                
                eval_dict(probs, label, self.prop_names,  IS_R=self.IS_R)
            
            if early_stop: print('early stop'); break

        print('Finished training\n')
        if len(data_df) > 0: # evaluated PCA on data_df, create gif
            for n in self.prop_names: 
                images = []
                for i in range(epoch+1):
                    file_name = self.figure_path + f'/PCA_{n}_{i}.png'
                    # print(file_name)
                    for j in range(5): 
                        try: images.append(imageio.imread(file_name))
                        except: pass
                gif_path = f'{self.figure_path}_{n}.gif'
                imageio.mimsave(gif_path, images, duration=1)
                print('save gif at: ', gif_path)
                # from IPython.display import Image
                display(IPython.display.Image(data=open(gif_path, 'rb').read(),
                        format='png'))

        
        self.save_train_status() # status yml is saved iff train finished
        # print('task weight', 
        #                 ['{:.3f}'.format(i) for i in self.weight_loss])
        print('Model parameters: ', count_parameters(self.model))
        self.get_runtime()
        print(f"best epoch: {self.best_epoch}, min loss: {self.min_loss:.4f}")
        plot_loss(self.train_dict, self.valid_dict, name='valid',
                title_name= f'loss during training {self.model_type}')
        plot_loss(train_cls, valid_cls, name='valid',
                title_name= f'Classify loss during training {self.model_type}')
        
        if test_loader != None: # evaluate test set
            self.performance_dict,_,_ = self.eval(test_loader, self.model_path)
            self.save_train_status() # update train status with test performance
        print('Finished evaluate test performance, outputs performance dict')
        return self.performance_dict

    def predict(self, smile_list:list, return_probs=False, scale_back=True):
        if self.model_type == 'VAE': self.load_vae_pretrain()
        else: self.load_model(self.model_path)
        if isinstance(smile_list, str): smile_list = [smile_list]
        df = pd.DataFrame(); df['Drug'] = smile_list 
        
        for name in self.prop_names:
            df[name] = [MASK] * len(smile_list)
        
        prm = {'batch_size': 64, 'shuffle': False, 
               'drop_last': False, 'num_workers': 0}
        loader = get_loader(df, self.prop_names, prm, self.model_type,
                            self.vocab, self.vocab_type)
        
        y_probs = model_predict(self.model, loader, self.IS_R, self.prop_names,
                self.device, self.model_type, MASK, self.scale_dict, scale_back)
        for name in y_probs.keys():
            is_r = names_dict[name]
            if is_r == False and return_probs==False: # cls, prob -> pred
                df[name] = get_preds(0.5, y_probs[name])
            else: df[name] = y_probs[name]
        return df

    def train(self, data_loader, val_loader, test_loader=None,
                 data_df=pd.DataFrame(), sample_num=1, latent_eval_freq=1): 
        if self.best_epoch != 0:
            self.model.load_state_dict(torch.load(
                self.model_path, map_location=self.device))
        else: print(f'Start training {self.model_type}...')
        # if 'MAX_EPOCH' not in self.config: MAX_EPOCH = 1000
        # else:          MAX_EPOCH = self.config['MAX_EPOCH']
        # single task, no need uncertainty weight
        if self.model_type == 'VAE': 
            return self.train_VAE(data_loader, val_loader, test_loader,
                                  data_df, sample_num, latent_eval_freq)

        if len(self.prop_names) == 1: self.uw = False 
        if self.uw: 
            m_w = MTLoss(len(self.prop_names), device=self.device) 
            optimizer = torch.optim.SGD(m_w.parameters(), lr=0.1); m_w.train()
    
        
        for epoch in range(self.best_epoch, self.MAX_EPOCH):
            t = time.time()
            score, l, r  = self.train_fn(self.model, data_loader, self.IS_R,
                                  self.prop_names, self.device, epoch,
                                  self.optimizer, scale_dict=self.scale_dict,
                                  weight_loss=self.weight_loss)
            train_time = (time.time() - t) * 1000 / len(data_loader.dataset)
            self.times_list.append(train_time)

            if self.uw: # uncertainty weight training
                optimizer.zero_grad()
                total_loss, self.weight_loss = m_w(l, r)
                total_loss.backward(); optimizer.step()
            # do not use weight loss for val?  
            val_score, probs, labels = self.train_fn(self.model,  val_loader,
                                       self.IS_R,self.prop_names,self.device,
                                       epoch,   scale_dict = self.scale_dict,
                                       weight_loss=self.weight_loss) 
            
            self.train_dict[epoch] = score
            self.valid_dict[epoch] = val_score
            print(f'Epoch:{epoch} [Train] Loss: {score:.3f} |',
                  f'[Valid] Loss: {val_score:.3f}', end="\t")
            early_stop = self.stopper.step(val_score, self.model)
            
            if val_score < self.min_loss: # loss drop, save model
                print(f'SAVE MODEL: loss: {self.min_loss:.3f} -> '
                      f'{val_score:.3f} | runtime: {train_time:.3f} ms')
                self.min_loss = val_score;  self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_path)

            if epoch % self.verbose_freq == 0 and epoch != 0:
                self.get_runtime()
                if self.uw:                 print('different task weight', 
                           ['{:.3f}'.format(i) for i in self.weight_loss])
                plot_loss(self.train_dict, self.valid_dict, name='valid',
                    title_name= f'loss during training {self.model_type}')
                eval_dict(probs, labels, self.prop_names,  IS_R=self.IS_R)
                
            if early_stop: print('early stop'); break

        print('Finished training\n')
        
        self.save_train_status() # status yml file is saved iff train finished
        print('task weight', 
                        ['{:.3f}'.format(i) for i in self.weight_loss])
        print('Model parameters: ', count_parameters(self.model))
        self.get_runtime()
        print(f"best epoch: {self.best_epoch}, min loss: {self.min_loss:.4f}")
        plot_loss(self.train_dict, self.valid_dict, name='valid',
                  title_name= f'loss during training {self.model_type}')
        
        if test_loader != None: # evaluate test set
            self.performance_dict,_,_ = self.eval(test_loader, self.model_path)
            self.save_train_status() # update status yml with test performance
        print('Finished evaluate test performance, outputs performance dict')
        return self.performance_dict



