from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def unzip_vals(this_accum, this_tag):
    w_times, step_nums, vals = zip(*this_accum.Scalars(this_tag))
    return w_times, step_nums, vals


def get_accumulator_data(cls_name, rootdir = '/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/'):
    event_acc = EventAccumulator(rootdir+ '%s/train_acc/acc_rgbd' % cls_name)
    event_loss_all = EventAccumulator(rootdir + '%s/loss/loss_all' % cls_name)
    event_loss_ctr = EventAccumulator(rootdir + '%s/loss/loss_ctr_of' % cls_name)
    event_loss_kp = EventAccumulator(rootdir + '%s/loss/loss_kp_of' % cls_name)
    event_loss_rgbd = EventAccumulator(rootdir + '%s/loss/loss_rgbd_seg' % cls_name)
    event_loss_target = EventAccumulator(rootdir + '%s/loss/loss_target' % cls_name)
    #event_val_acc = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/any2/val_acc/acc_rgbd')
    event_val_acc_avg = EventAccumulator(rootdir + '%s/val_acc_avg/acc_rgbd' % cls_name)

    event_acc.Reload()
    event_loss_all.Reload()
    event_loss_ctr.Reload()
    event_loss_kp.Reload()
    event_loss_rgbd.Reload()
    event_loss_target.Reload()
    #event_val_acc.Reload()
    event_val_acc_avg.Reload()
    # Show all tags in the log file
    print(event_acc.Tags())
    print(event_loss_all.Tags())
    #print(event_val_acc.Tags())
    print(event_val_acc_avg.Tags())
    #print(event_loss_ctr.Tags())
    #print(event_loss_kp.Tags())
    #print(event_loss_rgbd.Tags())
    #print(event_loss_target.Tags())

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    w_times, step_nums, vals = unzip_vals(event_acc, 'train_acc')
    w1, s1, v1 = unzip_vals(event_loss_rgbd, 'loss')
    wt, st, vt = unzip_vals(event_loss_target, 'loss')
    wa, sa, va = unzip_vals(event_loss_all, 'loss')
    # wv, sv, vv = unzip_vals(event_val_acc, 'val_acc')
    wv2, sv2, vv2 = unzip_vals(event_val_acc_avg, 'val_acc_avg')

    data = {}
    data['train_acc'] = [step_nums, vals]
    data['rgbd_loss'] = [s1, v1]
    data['target_loss'] = [st, vt]
    data['all_loss'] = [sa, va]
    data['val_acc_avg'] = [sv2, vv2]

    return data

def get_all_tag_data(cls_lst):





#print(len(wv), len(sv), len(vv))
print(len(wv2), len(sv2), len(vv2))

print(len(w_times), len(step_nums), len(vals))

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


