import os.path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from ffb6d.multilabel_common import Config
import numpy as np

config = Config(ds_name='linemod', cls_type='')

def unzip_vals(this_accum, this_tag):
    w_times, step_nums, vals = zip(*this_accum.Scalars(this_tag))
    return w_times, step_nums, vals


def get_accumulator_data_lm(mrootdir):  # = '/home/nachiket/Documents/saved_models/my_models/peregrine_models_AttFFB6D_run1/other_logs/train_info/'
    all_obj_data = {}
    for k in config.lm_obj_dict:
        if not os.path.isdir(mrootdir+'%s/'):
            continue
        cls_name = k
        rootdir = mrootdir
        event_acc = EventAccumulator(rootdir+ '%s/train_acc/acc_rgbd' % cls_name)
        event_loss_all = EventAccumulator(rootdir + '%s/loss/loss_all' % cls_name)
        event_loss_ctr = EventAccumulator(rootdir + '%s/loss/loss_ctr_of' % cls_name)
        event_loss_kp = EventAccumulator(rootdir + '%s/loss/loss_kp_of' % cls_name)
        event_loss_rgbd = EventAccumulator(rootdir + '%s/loss/loss_rgbd_seg' % cls_name)
        event_loss_target = EventAccumulator(rootdir + '%s/loss/loss_target' % cls_name)
        event_val_acc_avg = EventAccumulator(rootdir + '%s/val_acc_avg/acc_rgbd' % cls_name)
        event_val_acc_out = EventAccumulator(rootdir + cls_name+'/')

        event_acc.Reload()
        event_loss_all.Reload()
        event_loss_ctr.Reload()
        event_loss_kp.Reload()
        event_loss_rgbd.Reload()
        event_loss_target.Reload()
        event_val_acc_avg.Reload()
        event_val_acc_out.Reload()
        # Show all tags in the log file
        #print(event_acc.Tags())
        #print(event_loss_all.Tags())
        #print(event_val_acc.Tags())
        #print(event_val_acc_avg.Tags())
        #print(event_loss_ctr.Tags())
        #print(event_loss_kp.Tags())
        #print(event_loss_rgbd.Tags())
        #print(event_loss_target.Tags())

        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        w_times, step_nums, vals = unzip_vals(event_acc, 'train_acc')
        w1, s1, v1 = unzip_vals(event_loss_rgbd, 'loss')
        wt, st, vt = unzip_vals(event_loss_target, 'loss')
        wa, sa, va = unzip_vals(event_loss_all, 'loss')
        wv2, sv2, vv2 = unzip_vals(event_val_acc_avg, 'val_acc_avg')
        wv3, sv3, vv3 = unzip_vals(event_val_acc_out, 'val_acc')

        data = {}
        data['train_acc'] = [step_nums, vals]
        data['rgbd_loss'] = [s1, v1]
        data['target_loss'] = [st, vt]
        data['all_loss'] = [sa, va]
        data['val_acc_avg'] = [sv2, vv2]
        data['val_acc_out'] = [sv3, vv3]

        all_obj_data.update({cls_name:data})

    return all_obj_data

def get_all_tag_data(cls_lst):
    pass

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = max(y)
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)




rootdir = '/home/nachiket/Documents/saved_models/my_models/peregrine_models_AttFFB6D_run1/other_logs/train_info/'
rootdir_fuse_only = '/home/nachiket/Documents/saved_models/otjer/peregine_models_fused_and_attention_ATTFFB_run2/train_info/'
rootdir_attffb_run1 = '/home/nachiket/Documents/saved_models/otjer/peregrine_models_AttFFB6D_run1/other_logs/train_info/'
root_dir_deepvit_timeout1 = '/home/nachiket/Documents/saved_models/otjer/deepvit_pere1_half_timeout/train_info/'
#event_rootval = EventAccumulator('/home/nachiket/Documents/saved_models/my_models/peregrine_models_AttFFB6D_run1/other_logs/train_info/ape/')
#event_lossall = EventAccumulator('/home/nachiket/Documents/saved_models/my_models/peregrine_models_AttFFB6D_run1/other_logs/train_info/ape/loss/loss_all')
#event_acc = EventAccumulator('/home/nachiket/Documents/saved_models/my_models/peregrine_models_AttFFB6D_run1/other_logs/train_info/ape/train_acc/acc_rgbd/')
#event_rootval.Reload()
#event_lossall.Reload()
#event_acc.Reload()

#print(len(wv), len(sv), len(vv))

#alldat = get_accumulator_data_lm(rootdir_attffb_run1)
alldat = get_accumulator_data_lm(root_dir_deepvit_timeout1)

cls_color = [
    (254.0, 254.0, 254.0),  # 0
    (180.0, 105.0, 254.0),   # 194, 194, 0,    # 1 # 194, 194, 0
    (0.0, 254.0, 0.0),      # 2
    (0.0, 0.0, 254.0),      # 3
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
    (0.0, 194.0, 0.0),      # 14
    (254.0, 254.0, 0.0),    # 15 # 0, 194, 194
    (64.0, 64.0, 0.0),      # 16
    (64.0, 0.0, 64.0),      # 17
    (185.0, 218.0, 254.0),  # 0, 0, 64,       # 18
    (0.0, 0.0, 254.0),      # 19
    (0.0, 64.0, 0.0),       # 20
    (0.0, 0.0, 192.0)       # 21
 ]

import matplotlib.pyplot as plt


def get_all_loss_charts():
    for k in config.lm_obj_dict:
        thisdat = alldat.get(k)['all_loss']
        #print('%s: stp, val : ', k, len(thisdat[0]), len(thisdat[1]))
        stp = thisdat[0]
        val = thisdat[1]

        stp_avg = [sum(stp[i*50:(i+1)*50])/50 for i in range(0,200)]
        val_avg = [sum(val[i*50:(i+1)*50])/50 for i in range(0,200)]
        newstp = [x for x in range(0,len(val_avg))]

        valmax = min(val)
        valavgmin = min(val_avg)
        print('%s: min val : ', k, valavgmin)

        colortag = tuple([x/255.0 for x in list(cls_color[config.lm_obj_dict.get(k)])])
        plt.plot(newstp, val_avg, label=k, color=colortag) #stp[:50],

        #annot_max(newstp, val_avg, ax=plt)
        note = str(k) + ' :' + '{0:.4f}'.format(valavgmin)
        plt.annotate(note , xy=(newstp[np.argmin(val_avg)], min(val_avg)))

    plt.legend()
    plt.title('Total Loss')
    plt.ylabel('total_loss')
    plt.xlabel('step')
    plt.show()

def get_rgbd_loss_charts():
    for k in config.lm_obj_dict:
        thisdat = alldat.get(k)['rgbd_loss']
        #print('%s: stp, val : ', k, len(thisdat[0]), len(thisdat[1]))
        stp = thisdat[0]
        val = thisdat[1]

        stp_avg = [sum(stp[i*50:(i+1)*50])/50 for i in range(0,200)]
        val_avg = [sum(val[i*50:(i+1)*50])/50 for i in range(0,200)]
        newstp = [x for x in range(0,len(val_avg))]

        valmax = min(val)
        valavgmin = min(val_avg)
        print('%s: min val : ', k, valavgmin)

        colortag = tuple([x/255.0 for x in list(cls_color[config.lm_obj_dict.get(k)])])
        plt.plot(newstp, val_avg, label=k, color=colortag) #stp[:50],

        #annot_max(newstp, val_avg, ax=plt)
        note = str(k) + ' :' + '{0:.4f}'.format(valavgmin)
        plt.annotate(note , xy=(newstp[np.argmin(val_avg)], min(val_avg)))

    plt.legend()
    plt.title('RGBD Loss')
    plt.ylabel('rgbd_loss')
    plt.xlabel('step')
    plt.show()

def get_target_loss_charts():
    for k in config.lm_obj_dict:
        thisdat = alldat.get(k)['target_loss']
        #print('%s: stp, val : ', k, len(thisdat[0]), len(thisdat[1]))
        stp = thisdat[0]
        val = thisdat[1]

        stp_avg = [sum(stp[i*50:(i+1)*50])/50 for i in range(0,200)]
        val_avg = [sum(val[i*50:(i+1)*50])/50 for i in range(0,200)]
        newstp = [x for x in range(0,len(val_avg))]

        valmax = min(val)
        valavgmin = min(val_avg)
        print('%s: min val : ', k, valavgmin)

        colortag = tuple([x/255.0 for x in list(cls_color[config.lm_obj_dict.get(k)])])
        plt.plot(newstp, val_avg, label=k, color=colortag) #stp[:50],

        #annot_max(newstp, val_avg, ax=plt)
        note = str(k) + ' :' + '{0:.4f}'.format(valavgmin)
        plt.annotate(note , xy=(newstp[np.argmin(val_avg)], min(val_avg)))

    plt.legend()
    plt.title('Target Loss')
    plt.ylabel('target_loss')
    plt.xlabel('step')
    plt.show()


def get_val_acc_charts():
    for k in config.lm_obj_dict:
        thisdat = alldat.get(k)['val_acc_out']
        #print('%s: stp, val : ', k, len(thisdat[0]), len(thisdat[1]))
        stp = thisdat[0]
        val = thisdat[1]

        stp_avg = [sum(stp[i*50:(i+1)*50])/50 for i in range(0,200)]
        val_avg = [sum(val[i*50:(i+1)*50])/50 for i in range(0,200)]
        newstp = [x for x in range(0,len(val_avg))]

        valmax = max(val)
        valavgmax = max(val_avg)
        print('%s: max val : ', k, valavgmax)

        colortag = tuple([x/255.0 for x in list(cls_color[config.lm_obj_dict.get(k)])])
        plt.plot(newstp, val_avg, label=k, color=colortag) #stp[:50],

        #annot_max(newstp, val_avg, ax=plt)
        note = str(k) + ' :' + '{0:.4f}'.format(valavgmax)
        plt.annotate(note , xy=(newstp[np.argmax(val_avg)], max(val_avg)))

    plt.legend()
    plt.title('Validation Accuracy')
    plt.ylabel('validation_acc')
    plt.xlabel('step')
    plt.show()


def get_train_acc_charts():
    for k in config.lm_obj_dict:
        thisdat = alldat.get(k)['train_acc']
        # print('%s: stp, val : ', k, len(thisdat[0]), len(thisdat[1]))
        stp = thisdat[0]
        val = thisdat[1]

        print(len(stp))
        print(len(val))

        stp_avg = [sum(stp[i * 50:(i + 1) * 50]) / 50 for i in range(0, 200)]
        val_avg = [sum(val[i * 50:(i + 1) * 50]) / 50 for i in range(0, 200)]
        newstp = [x for x in range(0, len(val_avg))]

        valmax = max(val)
        valavgmax = max(val_avg)
        print('%s: max val : ', k, valavgmax)

        colortag = tuple([x / 255.0 for x in list(cls_color[config.lm_obj_dict.get(k)])])
        plt.plot(newstp, val_avg, label=k, color=colortag)  # stp[:50],

        # annot_max(newstp, val_avg, ax=plt)
        note = str(k) + ' :' + '{0:.4f}'.format(valavgmax)
        plt.annotate(note, xy=(newstp[np.argmax(val_avg)], max(val_avg)))

    plt.legend()
    plt.title('Training Accuracy')
    plt.ylabel('train_acc')
    plt.xlabel('step')
    plt.show()

get_train_acc_charts()
#get_val_acc_charts()
#get_all_loss_charts()
#get_rgbd_loss_charts()
#get_target_loss_charts()

'''
import matplotlib.pyplot as plt

# plot lines
plt.plot(step_nums[-267:], vals[-267:], label="train_acc")
plt.plot(step_nums[-267:], v1[-267:], label="rgbd_loss")
#plt.plot(step_nums[-380:], vt[-380:], label="target_loss")
#plt.plot(step_nums[-380:], va[-380:], label="total_loss")
plt.legend()
plt.show()


#plt.plot(sv, vv, label='val_acc')
plt.plot(sv2, vv2, label='val_acc_avg')
plt.legend()
plt.show()
'''

