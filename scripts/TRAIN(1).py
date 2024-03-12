from scripts.func_utils import *
from scripts.yaml_utils import *
from scripts.eval_utils import *
from scripts.preprocess_mols import *
from scripts.model_architecture import *
from scripts.dataset import *
from scripts.train import *

from tdc.utils import retrieve_label_name_list
import pandas as pd
from scripts.get_vocab import *

from tdc.single_pred import ADME
from tdc.single_pred import Tox
label_list = retrieve_label_name_list('herg_central')

import yaml


def train_mt(name, model_type, folder_name, repeat_time, retrain, uw, 
             clean_mol_=False):
    """
    Aim: train single task or multiple task, regression or classification
    param
        name:        str or list, name(s) of task
        model_type:  str, model type
        folder_name: str, the dir of saved model and yml files
        repeat_time: int, repeat training the same model several times
        retrain:     bool,
                     if False, will check whether there exists pretrained
                     else:     retrain anyway and delete past files
        uw:          bool, use uncertainty weight between tasks or not
    Return a list, each element is a dict, contain performance and train loss
    """
    # if len(name) == 1 or isinstance(name, str): # single task
    #     # return train_st(name, model_type, folder_name, repeat_time, retrain)
    #     run_type = 'ST'
    #     run_type = 'MT'
    if isinstance(name, str): name = [name]
    if len(name) == 1: run_type = 'ST'; uw = False  # single task, no need uncertainty_weight
    elif len(name) >1: run_type = 'MT'
    make_path(folder_name, False); perfs = []
    config = get_config(model_type, name)
    dataloader_ready = False
    for i in range(repeat_time):
        print(f'\nRun # {i} for {model_type} {run_type}', end='\t')
        save_dir = f'{folder_name}/{model_type}_{run_type}_{i}'
        print(' | save dir: ', save_dir, end=' | \t')
        config['model_path']  = save_dir + '.pt'
        config['config_path'] = save_dir + '.yml'
        config['uncertainty_weight'] = uw
        config['verbose_freq'] = 100
        config['patience'] = 30
        config['MAX_EPOCH'] = 1000
        if run_type == 'MT': config['lr'] = 1e-4

        nofile = False
        if retrain == False:
            try: # try open yml file, if file exists, and no need train
                with open(config['config_path'], 'r') as f:
                    data = yaml.safe_load(f)
                if data != None:
                    p = yml_report(data); print('--> pre data loaded')
                nofile = False
            except:
                print(f"cannot open {config['config_path']}, retrain")
                nofile = True # model was not trained yet, train the model
        if nofile or retrain: # there is no trained file saved or retrain anyway
            if dataloader_ready == False: # no dataloader, prepare it  
                trn, val, tst = collect_data(name, clean_mol_=clean_mol_)
                dict_scale = None; scale_here = False
                for n in name: # scale regression tasks
                    if names_dict[n] == True: scale_here = True; break
                trn, val, tst, dict_scale = scale(trn,val,tst, scale_here)
                config['scale_dict'] = dict_scale
                trn_l, val_l, tst_l, vocab = get_multi_loader(
                                                trn, val, tst, config)
                if vocab != None and config['vocab'] == None:
                    config['vocab'] = vocab # update config vocab info
                    print(f'RNN, update vocab using dataset | ',
                        f'vocab length updated:', len(vocab))
                dataloader_ready = True # finish dataloader preparation

            models = PRED(**config)
            p = models.train(trn_l, val_l, tst_l)

        # eval_perf_list(p, name, {});
        perfs.append(p)

    best_idx = eval_perf_list(perfs, name, {})
    eval_perf_list(perfs[best_idx], name)
    print('\n\n\n')
    return perfs
