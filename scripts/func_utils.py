"""
Date: 09/07/2023
Mods: 11/09/2023, added create_gif function
Athr: Bu
Aims: helper functions for coding and training process
Func:
      tbc...
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from os import walk
import os
from tdc import Oracle
qed = Oracle(name='QED')
sa = Oracle(name='SA')

def make_path(path_name, verbose=True):
    import os
    if os.path.exists(path_name):
        if verbose: print('path:', path_name, 'already exists')
    else: os.makedirs(path_name); print('path:', path_name, 'is created')

file_types = ['bin', 'pth']
# clean certain type of file in path
def clean_files(path='/content/drive/MyDrive/ADMET/', 
                file_types = ['pth', 'bin', 'pt', 'yml']):
    files = next(walk(path), (None, None, []))[2]
    for file_ in files:
        if isinstance(file_, str):
            file_type = file_.split('.')[-1]
            if file_type in file_types:
                file_here = path + file_
                os.remove(file_here); print(f'removed from {path}: {file_} ')


def create_gif(fig_path:str, name:list): 
    import imageio
    for n in name:
        images = []
        for i in range(0,100):
            for j in range(1):
                file_name = fig_path + f'/PCA_{n}_{i}.png'
                try:
                    images.append(imageio.imread(file_name))
                except: pass
        gif_path = f'{fig_path}_{n}.gif'
        imageio.mimsave(gif_path, images, duration=1)
        from IPython.display import Image
        print('---> load gif from ', gif_path)
        display(Image(data=open(gif_path,'rb').read(), format='png'))

def convert_with_qed_sa(train):
    smi_list = train['smiles'].tolist()
    smile_list = []
    qed_list = []
    sa_list = []
    for i, smi in tqdm(enumerate(smi_list), total=len(smi_list),
                       desc='cal QED/SA, delete invalid'):
        try:
            qed_ = qed(smi); sa_ = sa(smi)
            smile_list.append(smi)
            qed_list.append(qed_)
            sa_list.append(sa_)
        except: pass 
    df = pd.DataFrame()
    df['smiles'] = pd.DataFrame(smile_list)
    df['qed'] = pd.DataFrame(qed_list)
    df['sa'] = pd.DataFrame(sa_list)
    df = df.reset_index(drop=True)
    return df 

def get_min(d:dict):
    min_key = next(iter(d))
    for key in d: # Iterate over the keys in the dictionary
        # If the value of the current key > the value of max_key, update max_key
        if d[key] < d[min_key]: min_key = key
    return min_key, d[min_key]

def plot_loss(train_dict, test_dict, name='test', title_name=None):
    fig = plt.figure()
    # fig.grid(False)
    plt.plot(list(train_dict.keys()), list(train_dict.values()), label='train')
    plt.plot(list(test_dict.keys()), list(test_dict.values()), label=name)
    argmin, min_ = get_min(test_dict)
    plt.plot(argmin, min_, '*', label=f'min epoch {argmin}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if title_name == None: title_name = 'loss during training'
    plt.title(title_name); plt.grid(False)
    plt.legend()
    plt.show(); plt.close()

def plot_performance(list_of_dict, model_types, title=None): 
    
    # loss dict or performance dict
    assert len(list_of_dict) == len(model_types)
    fig = plt.figure()
    # fig.grid(False)
    for model_name, per in zip(model_types, list_of_dict):
        plt.grid(False)
        plt.plot(list(per.keys()), list(per.values()), label=model_name)
    plt.xlabel('epoch')
    plt.ylabel('performance')
    if title == None: title = 'Performance on valid set during training'
    plt.title(title); plt.grid(False)
    plt.legend()
    plt.show(); plt.close()
