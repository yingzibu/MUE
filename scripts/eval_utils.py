"""
Date: 09/07/2023
Mods: 
      11/08/2023: plot_dim_reduced: add ver, add different palette type
                  evaluate: edited MCC 
Athr: Bu
Aims: functions to evaluate model performance / dataset quality
Func:
      get_preds: proba in [0, 1] --> label 0 or 1
      roc_curve: classification task & plot AUROC
      prc_curve: classification task & plot AUPRC
      evaluate:  classification task eval metrics
      reg_evaluate:  regression task eval metrics
      eval_dict: evaluate tasks (both cls  & reg)
      roc_curve_batch:  plot multi AUROC in graph
      prc_curve_batch:  plot multi AUPRC in graph
      plot_dim_reduced: plot PCA/t-SNE for a task
"""

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from sklearn.metrics import confusion_matrix, roc_auc_score
import math
import sklearn.metrics as metrics
import numpy as np
from tdc import Evaluator
from mycolorpy import colorlist as mcp
from scripts.func_utils import make_path
eval_PCC = Evaluator(name = 'PCC')
eval_SCC = Evaluator(name = 'Spearman')

evaluate_names = ['ROC-AUC', 'PR-AUC']

def get_preds(thres, prob):
    try: 
        if prob.shape[1] == 2: prob = prob[:, 1]
    except: pass
    return [1 if p > thres else 0 for p in prob]

# AUC, AP figure generating
# code reference: 
# https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py

def roc_curve(y_pred, y_label, method_name, figure_title=None, figure_file=None):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
    '''
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)	
    fpr = dict()
    tpr = dict() 
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 2
    plt.plot(fpr[0], tpr[0],
            lw=lw, label= method_name + ' (area = %0.3f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    title = 'AUROC'
    if figure_title != None: title += ' on ' + figure_title + ' test set'
    plt.title(title, fontsize=fontsize)
    # plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    if figure_file != None: 
        plt.savefig(figure_file)
    plt.show(); plt.close()
    return 

# code reference: 
# https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py
def prc_curve(y_pred, y_label, method_name, figure_title=None, figure_file=None):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        reference: 
            https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    '''	
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
    #	plt.plot([0,1], [no_skill, no_skill], linestyle='--')
    label_name = ' (area = %0.3f)' % average_precision_score(y_label, y_pred)
    plt.plot(lr_recall, lr_precision, lw = 2, label= method_name+label_name)
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    # plt.title('Precision Recall Curve')
    title = 'PRAUC'
    if figure_title != None: title += ' on ' + figure_title + ' test set'
    plt.title(title, fontsize=fontsize)
    # plt.title('PRC', fontsize=fontsize)
    plt.legend()
    if figure_file != None:
        plt.savefig(figure_file)
    plt.show(); plt.close()
    return 


def evaluate(y_real, y_hat, y_prob, ver=True): # for classification 
    TN, FP, FN, TP = confusion_matrix(y_real, y_hat).ravel()
    ACCURACY = (TP + TN) / (TP + FP + TN + FN)
    SE = TP / (TP + FN); recall = SE; SP = TN / (TN + FP)
    weighted_accuracy = (SE + SP) / 2
    precision = TP / (TP + FP); SP = TN / (TN + FP)
    F1 = 2 * precision * recall /(precision + recall)

    temp = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)

    # MCC = (TP * TN - FP * FN) * 1.0 / (math.sqrt(temp)) if temp != 0 else np.nan
    if temp != 0: MCC = (TP * TN - FP * FN) * 1.0 / (math.sqrt(temp))
    else: MCC = np.nan
    # else:
    #     print('TP, FP, TN, FN: ', TP, FP, TN, FN, end=' | '); MCC = np.nan
    #     print('at least 2 of TP, FP, TN, FN are 0, cannot cal MCC')
    try:
        if y_prob.shape[1] == 2: proba = y_prob[:, 1]
        else: proba = y_prob
    except: proba = y_prob
    AP  = average_precision_score(y_real, proba)
    AUC = roc_auc_score(y_real, proba)
    # print(f'Accuracy, w_acc,   prec, recall/SE,   SP,   ',
    #       f'F1,     AUC,     MCC,     AP')
    if ver == True: 
        print(f'  Acc,  w_acc,   prec,  recall,   SP,   ',
              f' F1,    AUC,   MCC,   AP')
        
        print("&%5.3f"%(ACCURACY), " &%5.3f"%(weighted_accuracy), 
              " &%5.3f"%(precision), " &%5.3f"%(SE), " &%5.3f"%(SP), 
              " &%5.3f"%(F1), "&%5.3f"%(AUC), "&%5.3f"%(MCC), "&%5.3f"%(AP))
    # print(type(F1))
    return [ACCURACY, weighted_accuracy, precision, SE, SP, F1, AUC, MCC, AP]


def reg_evaluate(label_clean, preds_clean, ver=True):
    mae = metrics.mean_absolute_error(label_clean, preds_clean)
    mse = metrics.mean_squared_error(label_clean, preds_clean)
    rmse = np.sqrt(mse) #mse**(0.5)
    r2 = metrics.r2_score(label_clean, preds_clean)
    pcc = eval_PCC(label_clean, preds_clean)
    scc = eval_SCC(label_clean, preds_clean)

    if ver: print('  MAE     MSE     RMSE    R2     pcc     spearman')
    if ver: print("&%5.3f" % (mae), " &%5.3f" % (mse), " &%5.3f" % (rmse),
                    " &%5.3f" % (r2), " &%5.3f" % (pcc), " &%5.3f" % (scc))
    # return r2, mae, rmse
    return mae, mse, rmse, r2, pcc, scc


def eval_dict(y_probs:dict, y_label:dict, names:list, IS_R, model_type='model',
              draw_fig=False, fig_title=None, fig_path=None):
    """
    Return a dictionary of name: performance
    IS_R == True: regression task, returns R2
    IS_R == False: classific task, returns accuracy
    """
    
    if isinstance(names, str): names = [names]
    if isinstance(IS_R, list): task_list = IS_R
    else: task_list = [IS_R] * len(names)
    performances = {}
    for i, (name, IS_R) in enumerate(zip(names, task_list)):
        # IS_R = task_list[i]
        print('*'*15, name, '*'*15)
        # print('Regression task', IS_R)
        # print(y_probs)
        probs = y_probs[name]
        label = y_label[name]
        assert len(probs) == len(label)
        if IS_R == False: # classification task
            preds = get_preds(0.5, probs)
            cls_results = evaluate(label, preds, probs)
            if draw_fig:
                plt.grid(False)
                roc_curve(probs, label, model_type, figure_title=name)
                prc_curve(probs, label, model_type, figure_title=name)
            # performances[name] = float(cls_results[0]) # accuracy 
            performances[name] = [float(r) for r in cls_results]

        else: # regression task
            mae, mse, rmse, r2, pcc, scc = reg_evaluate(label, probs)
            # performances[name] = float(r2) # r2 
            performances[name]=[float(mae), float(mse), float(rmse), float(r2),
                                float(pcc), float(scc)]
            if draw_fig:
                plt.grid(False)
                color = mcp.gen_color_normalized(cmap='viridis', data_arr=label)
                plt.scatter(label, probs, cmap='viridis', marker='.',
                            s=10, alpha=0.5, edgecolors='none', c=color)
                plt.xlabel(f'True value'); plt.ylabel(f'Predicted value')
                if fig_title == None: 
                    title = f'{name} test set performance of {model_type}'
                else: title = f'{name} {fig_title}'
                plt.title(title)
                x0, xmax = plt.xlim();  y0, ymax = plt.ylim()
                data_width = xmax - x0; data_height = ymax - y0
                # print(x0, xmax, y0, ymax, data_width, data_height)
                r2   = f'R2:     {r2:.3f}'
                mae  = f'MAE:   {mae:.3f}'
                rmse = f'RMSE: {rmse:.3f}'
                plt.text(x0 + 0.1*data_width, y0 + data_height * 0.8/0.95, r2)
                plt.text(x0 + 0.1*data_width, y0 + data_height * 0.8,  mae)
                plt.text(x0 + 0.1*data_width, y0 + data_height * 0.8*0.95, rmse)
                if fig_path != None: # save figure at fig_path
                    make_path(fig_path, False); 
                    fig_name = f'{fig_path}/{title}.png'
                    plt.savefig(fig_name, format='png', transparent=False)

                plt.show(); plt.cla(); plt.clf(); plt.close()
        print()
    return performances


def roc_curve_batch(probs_list, label_list, model_types, 
                    figure_title=None, figure_file=None):

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    assert len(probs_list) == len(label_list)
    assert len(probs_list) == len(model_types)
    fig = plt.figure()
    for y_pred, y_label, method_name in zip(probs_list, label_list, model_types): 
        y_label = np.array(y_label)
        y_pred =  np.array(y_pred)	
        fpr = dict()
        tpr = dict() 
        roc_auc = dict()
        fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
        roc_auc[0] = auc(fpr[0], tpr[0])
        lw = 2
        plt.plot(fpr[0], tpr[0],
                lw=lw, label= method_name + ' (area = %0.3f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    title = 'AUROC'
    if figure_title != None: title += ' on ' + figure_title + ' test set'
    plt.title(title, fontsize=fontsize)
    # plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    if figure_file != None: 
        plt.savefig(figure_file)
    plt.show(); plt.close()
    return 

def prc_curve_batch(probs_list, label_list, model_types, 
                    figure_title=None, figure_file=None):

    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    assert len(probs_list) == len(label_list)
    assert len(probs_list) == len(model_types)
    fig = plt.figure()
    for y_pred, y_label, method_name in zip(probs_list, label_list, model_types): 
        lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
        label_name = ' (area = %0.3f)' % average_precision_score(y_label, y_pred)
        plt.plot(lr_recall, lr_precision, lw = 2, label= method_name+label_name)
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    # plt.title('Precision Recall Curve')
    title = 'PRAUC'
    if figure_title != None: title += ' on ' + figure_title + ' test set'
    plt.title(title, fontsize=fontsize)
    # plt.title('PRC', fontsize=fontsize)
    plt.legend()
    if figure_file != None:
        plt.savefig(figure_file)
    plt.show(); plt.close()
    return 



def plot_dim_reduced(mol_info, label, task_type, dim_reduct='PCA',
                     title=None, savepath=None, savename=None, ver=False, 
                     plot_show=True):
    """
    param mol_info: could be MACCS Fingerprint
    param label: label of data
    param task_type: [True, False], True:regression; False: classification
    param dim_reduct" : ['PCA', 't-SNE']
    param title: None or string, the name of the plot
    Return figure.png saved at dim_reduct/title.png
    """
    features, labels = mol_info.copy(), label.copy()
    n_components = 2
    if dim_reduct == 'PCA':
        pca = PCA(n_components=n_components)
        pca.fit(features)
        features = StandardScaler().fit_transform(features)
        features = pd.DataFrame(data = pca.transform(features))
        ax_label = 'principle component'
    elif dim_reduct=='t-SNE':
        features = TSNE(n_components=n_components).fit_transform(features)
        features = MinMaxScaler().fit_transform(features)
        features = pd.DataFrame(np.transpose((features[:,0],features[:,1])))
        ax_label = 't-SNE'
    else: print("""Error! dim_reduct should be 'PCA' or 't-SNE'"""); return

    columns = [f'{ax_label} {i+1}' for i in range(n_components)]
    # features = pd.DataFrame(data = pca.transform(features), columns=columns)
    features.columns = columns
    features['label'] = labels

    sns.set_theme(style="whitegrid")
    # f, ax = plt.subplots(figsize=(6, 6))
    f, ax = plt.subplots()

    # vlag, Spectral, coolwarm
    if task_type == False: palette_type = 'vlag'
    else:                  palette_type = 'RdBu'
    param_dict = {'x': columns[0],
                'y': columns[1],
                'hue':'label',
                'palette': palette_type,
                'data': features,
                's': 10,
                'ax':ax}

    # sns.despine(f, left=True, bottom=False)
    sns.scatterplot(**param_dict)

    if task_type == True: # regression task, color bar for labels
        norm = plt.Normalize(labels.min(), labels.max())
        scalarmap = plt.cm.ScalarMappable(cmap=param_dict['palette'], norm=norm)
        scalarmap.set_array([])
        ax.figure.colorbar(scalarmap)
        ax.get_legend().remove()
    else: sns.move_legend(ax, 'upper right') # for classification, label box

    ax = plt.gca()
    # Set the border or outline color and width
    border_color = 'black'
    border_width = 0.6  # Adjust this as needed

    # Add a rectangular border around the plot
    for i in ['top', 'right', 'bottom', 'left']: ax.spines[i].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(border_width); spine.set_color(border_color)
    # move the legend if has that:

    if title == None: title = f'{dim_reduct}_demo'
    plt.title(title)
    if savepath != None:
        make_path(savepath, False)
        if savename == None: savename = f'{savepath}/{title}.png'
        else: savename = savepath + '/' + savename
        plt.savefig(savename, format='png', transparent=False)
        if ver: print(f'figure saved at {savename}')
    if plot_show: plt.show()
    plt.close()
     

"""TEST CODE"""
# name = 'HIA_Hou'; IS_R = False
# # name = 'Caco2_Wang'; IS_R = True
# trains, valids, tests = collect_data_10_24([name], show_dist=False)
# df_all   = pd.concat([trains, valids, tests], ignore_index=True, axis=0)
# data_list = [trains, valids, tests, df_all]
# desc_list = ['train set', 'valid set', 'test set', 'data set']
# for (data, desc) in zip(data_list, desc_list):
#     data = process(data)
#     features, labels = data[header], data[name]
#     assert features.shape[0] == len(labels)
#     for dim_reduct in ['PCA', 't-SNE']:
#         title = f'{dim_reduct} on {desc} of {name}'
#         if dim_reduct == 'PCA':
#             plot_dim_reduced(features, labels, IS_R, dim_reduct, title)
