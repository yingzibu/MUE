"""
Date: 09/07/2023
Mods: 01/25/2024, add ENSEMBLE
      11/07/2023, edited get_config: config_RNN_pretrain
Athr: Bu
Aims: globally used constants
Func:
      get_config: return proper config for initialize models
"""


### CONSTANTS ###

prm = {'batch_size': 64, 'shuffle': False, 
       'drop_last': False, 'num_workers': 0}
cls_metrics = ['acc', 'w_acc', 'prec', 'recall', 'sp', 'f1', 'auc', 'mcc', 'ap']
reg_metrics = ['mae', 'mse', 'rmse', 'r2', 'pcc', 'spearman']

d = {'reg': [0,   2,    3], 'cls': [0,   6,   8]}
#            mae, rmse, r2          acc, auc, ap

MACCS_LEN = 167
header = ['bit' + str(i) for i in range(MACCS_LEN)]

MORGAN_LEN = 2048
header_MORGAN = ['bit'+str(i) for i in range(MORGAN_LEN)]
RADIUS = 2 # Morgan / ECFP4
# RADIUS = 3 # ECFP6


names_reg = ['Caco2_Wang', 'Lipophilicity_AstraZeneca',
         'HydrationFreeEnergy_FreeSolv', 'Solubility_AqSolDB', 'LD50_Zhu', 'Kp',
         'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ',
         'PPBR_AZ', 'VDss_Lombardo'] # regression task

names_cls = ['CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
             'CYP1A2_Veith', 'CYP2C9_Veith'] + \
            ['BBB_Martins', 'Bioavailability_Ma',
             'Pgp_Broccatelli', 'HIA_Hou','PAMPA_NCATS'] + \
            ['hERG_Karim', 'AMES'] + \
            ['CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels',
            'CYP3A4_Substrate_CarbonMangels', 
            'DILI', 'Skin Reaction','Carcinogens_Lagunin', 'ClinTox']

names_dict = {}
for name_ in names_reg + names_cls + ['qed', 'sa']:
    if name_ in names_reg + ['qed', 'sa']: names_dict[name_] = True  # regression task
    elif name_ in names_cls: names_dict[name_] = False # classification task

names_M5 = ['CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
            'CYP1A2_Veith', 'CYP2C9_Veith']
names_E3 = ['Half_Life_Obach', 
            'Clearance_Hepatocyte_AZ', 
            'Clearance_Microsome_AZ'] # 3 excretion tasks
            
names_T3 = ['hERG_Karim', 'AMES', 'LD50_Zhu']
names_A3 = ['PAMPA_NCATS', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB']
names_AD = ['BBB_Martins','PAMPA_NCATS', 
            'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB']
names_D3 = ['BBB_Martins', 'PPBR_AZ', 'VDss_Lombardo']

names_all = list(names_dict.keys())

# model_types = ['MLP', 'AttentiveFP', 'GIN', 'RNN']
model_types = ['MLP', 'AttentiveFP', 'GIN', 'RNN', 'VAE', 'RNN_pretrain']
VOCAB_TYPES = ['char', 'smiles', 'selfies']
# VOCAB_TYPE = 'selfies' # choose from VOCAB_TYPES
VOCAB_TYPE = 'smiles'
MASK = -100

pretrain_encoder_path = 'VAE/qed_sa_encoder.pt'
pretrain_decoder_path = 'VAE/qed_sa_decoder.pt'

import yaml
def load_vocab(VOCAB_TYPE):
    try:
        with open(f'vocab/{VOCAB_TYPE}.yml', 'r') as f: data = yaml.safe_load(f)
        vocab = data['vocab']; assert VOCAB_TYPE == data['vocab_type']
    except: vocab = None
    return vocab
# print('Vocab type for RNN:', VOCAB_TYPE)
# VOCAB = load_vocab(VOCAB_TYPE)


# universal constant for model training 
scale_dict = None
uw = False # uncertainty weight

dropout = 0.1
lr = 3e-4
wd = 1e-5
MAX_EPOCH = 1000
patience = 30           # stop if loss no decrease after epochs # patience
verbose_frequency = 100 # print evaluation every # verbose_frequency epoch
batch_size = 128

# special for MLP
MLP_in_dim = MACCS_LEN     # len of MACCS fingerprints
MLP_hid_dims = [128, 64, 32, 16]   # hidden dims 

# special for AttentiveFP
graph_feat_size = 300
n_layers = 5
dropout_ATF = 0.5
num_timesteps = 1   # times of updating the graph representations with GRU

# special for GIN: pretrain model types for selection:
pre_models_GIN = ['gin_supervised_contextpred', 'gin_supervised_infomax',
                     'gin_supervised_edgepred', 'gin_supervised_masking']
pre_model_num = 0    # choose from pre_models for GIN
in_dim = 256
hid_dims = [128, 64, 32, 16]

# if VOCAB_TYPE == 'smiles':



# special for RNN:
Bid = True
GRU_num_layers = 3
GRU_dim = 256
RNN_hid_dims = [128]
RNN_dropout = 0.5

# special for VAE: 
z_dim = 256
decoder_hid_dim = 512
decoder_hid_dims = [256, 512]
decoder_dropout = 0.1
decoder_num_layers = 3
cls_weight = 1
max_kl_weight = 0.5
 

def get_config(model_type, names,
               pre_model_num=pre_model_num, 
               scale_dict=scale_dict, 
               vocab_type=VOCAB_TYPE, 
               IS_R_default=False):
    """
    Get config to initialize model
        param model_type: str, ['MLP', 'AttentiveFP', 'GIN', 'RNN', 'MUE']
        param names: list, task names
        param scale_dict: dict,
            if the task is regression, could scale label values
                            {name: [value_min, value_max], ...}
        param pre_model_num: int, [0, 1, 2, 3]
            if model_type is 'GIN', 4 types of pretrained models to choose from
    Returns config that could be used as PRED(**config)
    """
    pre_models_GIN = ['gin_supervised_contextpred', 'gin_supervised_infomax',
                         'gin_supervised_edgepred', 'gin_supervised_masking']

    # print(scale_dict)
    
    if isinstance(names, str): names = [names]
    try: IS_R = [names_dict[name] for name in names]
    except: 
        print(f'{names} not in names_dict, assume reg task: {IS_R_default}')
        IS_R = [IS_R_default] * len(names)

    config_MLP = {'model_type': 'MLP',
            'in_dim': MLP_in_dim,
            'hid_dims': MLP_hid_dims,
            'out_dim': len(names),
            'prop_names': names,
            'dropout': dropout,
            'IS_R': IS_R,
            'batch_size': batch_size,
            'lr': lr,
            'wd': wd,
            'patience': patience,
            'verbose_freq': verbose_frequency,
            'model_path': f'ckpt_MLP.pt',
            'scale_dict': scale_dict}
    
    config_ENS = {'model_type': 'MUE',
            'in_dim': len(['MLP', 'AttentiveFP', 'GIN', 'RNN']), # ensemble of 4 models
            'out_dim': 1,
            'hid_dims': [128, 64, 32, 16],
            'dropout': 0.1,
            'prop_names': names,
            'IS_R': IS_R,
            'batch_size': batch_size,
            'lr': 3e-5,
            'wd': wd,
            'patience': 30,
            'verbose_freq': 50,
            'model_path': f'ckpt_MUE.pt',
            # 'weight_loss': [1.0],
            # 'uncertainty_weight': False,
            'scale_dict': None}

    config_ATF = {'model_type': 'AttentiveFP',
            'graph_feat_size': graph_feat_size,
            'num_timesteps': num_timesteps,
            'n_layers': n_layers,
            'out_dim': len(names),
            'prop_names': names,
            'dropout': dropout_ATF,
            'IS_R': IS_R,
            'batch_size': batch_size,
            'lr': lr,
            'wd': wd,
            'patience': patience,
            'verbose_freq': verbose_frequency,
            'model_path': 'ckpt_AT.pt',
            'scale_dict': scale_dict}

    config_GIN = {'model_type': 'GIN',
            'pretrain_model': pre_models_GIN[pre_model_num],
            'in_dim': in_dim,
            'hid_dims': hid_dims,
            'out_dim': len(names),
            'prop_names': names,
            'dropout': dropout,
            'batch_size': batch_size,
            'IS_R': IS_R,
            'lr': lr,
            'wd': wd,
            'patience': patience,
            'verbose_freq': verbose_frequency,
            'model_path': f'ckpt_GIN_{pre_models_GIN[pre_model_num]}.pt',
            'scale_dict': scale_dict}
    
    vocab = load_vocab(vocab_type)
    config_RNN = {'model_type': 'RNN',
              'vocab': vocab,
              'vocab_type': vocab_type,
              'Bidirect': Bid,
              'num_layers': GRU_num_layers,
              'GRU_dim': GRU_dim,
              'hid_dims': RNN_hid_dims,
              'out_dim': len(names),
              'prop_names': names,
              'dropout': RNN_dropout,
              'IS_R': IS_R,
              'device': 'cuda',
              'batch_size': batch_size,
              'lr': lr,
              'wd': wd,
              'patience': patience,
              'verbose_freq': verbose_frequency,
              'model_path': f'ckpt_RNN_{VOCAB_TYPE}.pt',
              'scale_dict': scale_dict}

    config_VAE = {'model_type': 'VAE',
              'vocab': vocab,
              'vocab_type': vocab_type,
              'Bidirect': Bid,
              'num_layers': GRU_num_layers,
              'GRU_dim': GRU_dim,
              'z_dim': z_dim,
              'dropout': RNN_dropout,
              'decoder_hid_dim': decoder_hid_dim,
              'decoder_num_layers': decoder_num_layers,
              'decoder_dropout': decoder_dropout,
              'decoder_hid_dims': decoder_hid_dims,
              'MLP_hid_dims': MLP_hid_dims,
              'MLP_dropout': dropout,
              'out_dim': len(names),
              'prop_names': names,
              'IS_R': IS_R,
              'device': 'cuda',
              'batch_size': batch_size,
              'lr': lr,
              'wd': wd,
              'cls_weight': cls_weight,
              'max_kl_weight': max_kl_weight,
              'patience': patience,
              'verbose_freq': verbose_frequency,
              'model_path': f'ckpt_VAE_{vocab_type}.pt', 
              'encoder_path': f'ckpt_encoder_{vocab_type}.pt',
              'decoder_path': f'ckpt_decoder_{vocab_type}.pt',
              'classifier_path': f'ckpt_classifier_{vocab_type}.pt',
              'scale_dict': scale_dict}

    config_RNN_pretrain = {
        'model_type': 'RNN_pretrain',
        'vocab': vocab,
        'vocab_type': vocab_type,
        'Bidirect': Bid,
        'num_layers': GRU_num_layers,
        'GRU_dim': GRU_dim,
        'z_dim': 256,
        'dropout': RNN_dropout,
        'MLP_hid_dims': MLP_hid_dims,
        'MLP_dropout': dropout,
        'out_dim': len(names),
        'prop_names': names,
        'IS_R': IS_R,
        'device': 'cuda',
        'batch_size': batch_size,
        'lr': lr,
        'wd': wd,
        'patience': patience,
        'verbose_freq': verbose_frequency,
        'model_path': f'ckpt_RNN_pretrain_{vocab_type}.pt', 
        'encoder_path': f'ckpt_RNN_pretrain_encoder_{vocab_type}.pt',
        'classifier_path': f'ckpt_RNN_pretrain_classifier_{vocab_type}.pt',
        'scale_dict': scale_dict
    }

    
    if   model_type == 'MLP':          con_MO = config_MLP
    elif model_type == 'AttentiveFP':  con_MO = config_ATF
    elif model_type == 'GIN':          con_MO = config_GIN
    elif model_type == 'RNN':          con_MO = config_RNN
    elif model_type == 'VAE':          con_MO = config_VAE
    elif model_type == 'MUE':          con_MO = config_ENS
    elif model_type == 'RNN_pretrain': con_MO = config_RNN_pretrain

    else: 
        print('Not in {MLP, AttentiveFP, GIN, RNN, VAE, RNN_pretrain, MUE}')
        return

    con_MO['config_path'] = con_MO['model_path'].split('.')[0] + '.yml'
    # different weight of task, initial weight the same
    con_MO['weight_loss'] = [float(1.0)/len(names)] * len(names)
    con_MO['MAX_EPOCH'] = MAX_EPOCH
    con_MO['uncertainty_weight'] = uw
    return con_MO

