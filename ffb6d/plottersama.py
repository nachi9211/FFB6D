import os.path

import pandas as pd
from ffb6d.multilabel_common import Config
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

lmconfig = Config(ds_name='linemod', cls_type='')
ycbconfig = Config(ds_name='ycb', cls_type='')

cls_color = [
    (254.0, 254.0, 254.0),  # 0
    (180.0, 105.0, 254.0),   # 194, 194, 0,    # 1 # 194, 194, 0
    (0.0, 254.0, 0.0),      # 2
    (0.0, 70.0, 154.0),      # 3
    (0.0, 254.0, 254.0),    # 4
    (254.0, 0.0, 254.0),    # 5
    (180.0, 105.0, 254.0),  # 128, 128, 0,    # 6
    (128.0, 0.0, 0.0),      # 7
    (0.0, 128.0, 0.0),      # 8
    (0.0, 165.0, 254.0),    # 0, 0, 128,      # 9
    (128.0, 128.0, 0.0),    # 10
    (0.0, 0.0, 254.0),      # 11
    (254.0, 0.0, 0.0),      # 12
    (0.0, 194.0, 0.0),      # 13
    (50.0, 94.0, 10.0),      # 14
    (254.0, 254.0, 0.0),    # 15 # 0, 194, 194
    (64.0, 64.0, 0.0),      # 16
    (64.0, 0.0, 64.0),      # 17
    (185.0, 218.0, 254.0),  # 0, 0, 64,       # 18
    (29.0, 100.0, 254.0),      # 19
    (0.0, 64.0, 0.0),       # 20
    (0.0, 0.0, 192.0)       # 21
 ]

def read_dfs_lm(base_dir, nametag):
    alldf = {}
    for k in lmconfig.lm_obj_dict:
        fname = nametag+'_'+str(k)+'_trainlog_df.csv'
        maindir = os.path.join(base_dir,fname)
        df = pd.read_csv(maindir)
        alldf.update({k:df})
        #Unnamed: 0, step_nums, train_acc, rgbd_loss, target_loss, all_loss, val_acc_avg, val_acc_out

        '''
        fname_val = nametag + '_VAL_' + str(k) + '_VAL_df.csv'
        valdir = os.path.join(base_dir,fname_val)
        valdf = pd.read_csv(valdir)
        '''

    return alldf



def read_dfs_ycb(base_dir, nametag):
    fname = nametag + '_trainlog_df.csv'
    maindir = os.path.join(base_dir, fname)
    df = pd.read_csv(maindir)
    # Unnamed: 0, step_nums, train_acc, rgbd_loss, target_loss, all_loss, val_acc_avg, val_acc_out


    return df



'AttFFB_RS'
'DeepAttFFB_deadline'
'DeepAttFFB_YCB'
'AttFFB_YCB'
'AttFFB_YCB_RS'

rootdir_attffb_run1 = '/home/nachiket/Documents/saved_models/otjer/peregrine_models_AttFFB6D_run1/other_logs/train_info/'
base_lm = read_dfs_lm(rootdir_attffb_run1, nametag='AttFFB')

rootdir_fuse_only = '/home/nachiket/Documents/saved_models/otjer/peregine_models_fused_and_attention_ATTFFB_run2/train_info/'
rs_lm = read_dfs_lm(rootdir_fuse_only, nametag='AttFFB_RS')

root_dir_deepvit_deadline = '/home/nachiket/Documents/saved_models/otjer/linemod_deepvit_basic_deadlinerush/train_info/'
deep_lm = read_dfs_lm(root_dir_deepvit_deadline, nametag='DeepAttFFB_deadline')

rootdir_YCB_deepvit = '/home/nachiket/Documents/saved_models/otjer/YCB_BASE_DeepViT/ycb/train_info/'
deep_ycb = read_dfs_ycb(rootdir_YCB_deepvit ,nametag='DeepAttFFB_YCB')

root_dir_YCB_base2 = '/home/nachiket/Documents/saved_models/otjer/YCB_PERE_BASE2/ycb/train_info/'
base_ycb = read_dfs_ycb(root_dir_YCB_base2, nametag='AttFFB_YCB')

rootdir_YCB_resupply2 = '/home/nachiket/Documents/saved_models/otjer/YCB_Resupply_Base_2/ycb/train_info/'
rs_ycb = read_dfs_ycb(rootdir_YCB_resupply2, nametag='AttFFB_YCB_RS')

#####
root_dir_deepvit_timeout1 = '/home/nachiket/Documents/saved_models/otjer/deepvit_pere1_half_timeout/train_info/'
root_dir_YCB_base = '/home/nachiket/Documents/saved_models/otjer/YCB_PERE_BASE/ycb/train_info/'
rootdir_YCB_resupply = '/home/nachiket/Documents/saved_models/otjer/YCB_Resupply_Base/ycb/train_info/'



def draw_graph_lm(df, pltkey, xlabel, ylabel, title, desc):
    step_nums = df.get('ape')['step_nums']
    for k in df.keys():
        this_df = df[k]

        colortag = tuple([x / 255.0 for x in list(cls_color[lmconfig.lm_obj_dict.get(k)])])
        plt.plot(step_nums, df[pltkey], label=k, color=colortag)  # stp[:50],


        #note = str(k) + ' :' + '{0:.4f}'.format(valavgmin)
        #plt.annotate(note, xy=(newstp[np.argmin(val_avg)], min(val_avg)))

    plt.legend(loc='best')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

    '''
    ['step_nums']
    ['train_acc']
    ['rgbd_loss']
    ['target_loss']
    ['all_loss']
    ['val_acc_out']
    '''
    return

def draw_graph_ycb(df, pltkey, xlabel, ylabel, title, desc):
    return

