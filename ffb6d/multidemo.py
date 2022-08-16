#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from common import Config, ConfigRandLA, ConfigTrans
from models.super_original_ffb6d import FFB6D
#from models.ffb6d import FFB6D
#from models.attentionffb import AttFFB6D as FFB6D
from datasets.ycb.ycb_dataset import Dataset as YCB_Dataset
from datasets.linemod.linemod_dataset import Dataset as LM_Dataset
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils
import pandas as pd
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-dataset", type=str, default="linemod",
    help="Target dataset, ycb or linemod. (linemod as default)."
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
parser.add_argument(
    "-show", action='store_true', help="View from imshow or not."
)
args = parser.parse_args()

if args.dataset == "ycb":
    config = Config(ds_name=args.dataset)
else:
    config = Config(ds_name=args.dataset, cls_type=args.cls)
bs_utils = Basic_Utils(config)


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint['model_state']
        if 'module' in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec


def cal_view_pred_pose(model, data, epoch=0, obj_id=-1):
    bgr_list = []
    ori_bgr_list = []
    pts_list = []
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)
        end_points_df = pd.DataFrame(end_points.items())
        #print(end_points_df.columns)
        #end_points_df.to_csv('~/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/endpoints_df_attffb.csv')
        _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        if args.dataset == "ycb":
            pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, True,
                None, None
            )
        else:
            pred_pose_lst = cal_frame_poses_lm(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, False, obj_id
            )
            pred_cls_ids = np.array([[1]])

        np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        if args.dataset == "ycb":
            np_rgb = np_rgb[:, :, ::-1].copy()
        ori_rgb = np_rgb.copy()
        for cls_id in cu_dt['cls_ids'][0].cpu().numpy():
            idx = np.where(pred_cls_ids == cls_id)[0]
            #print('Nachi: Generic cls_id: ',cls_id)   #1s
            if len(idx) == 0:
                #print('Nachi: Failed to find cls_id: ',cls_id)     #0s
                continue
            pose = pred_pose_lst[idx[0]]
            if args.dataset == "ycb":
                obj_id = int(cls_id[0])
            mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type=args.dataset).copy()
            mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
            #nachi
            #maxdist = mesh_pts
            #print('Nachi mesh points diff: ', maxdist)
            if args.dataset == "ycb":
                K = config.intrinsic_matrix["ycb_K1"]
            else:
                K = config.intrinsic_matrix["linemod"]
            mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
            color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
            np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
            #imshow('mid_loop_{}'.format(cls_id), np_rgb)
            #waitKey()
        vis_dir = os.path.join(config.log_eval_dir, "pose_vis")
        ensure_fd(vis_dir)
        f_pth = os.path.join(vis_dir, "{}.jpg".format(epoch))
        if args.dataset == 'ycb':
            bgr = np_rgb
            ori_bgr = ori_rgb
        else:
            bgr = np_rgb[:, :, ::-1]
            ori_bgr = ori_rgb[:, :, ::-1]
        cv2.imwrite(f_pth, bgr)
        temp_img = bgr - ori_bgr

        #Nachi: check if temp is sprayed, meaning no object ctr
        temp_intensity_mat = temp_img.sum(axis=2)
        temp_non_zeros = np.nonzero(temp_intensity_mat)  #tuples, first shouldnt start with zeroes.
        #print(temp_non_zeros)
        if temp_non_zeros[0][1]==0:
            temp_img = np.zeros(temp_img.shape)


        bgr_list.append(bgr)
        ori_bgr_list.append(ori_bgr)

        #if args.show:
        #    imshow("projected_pose_rgb", temp_img)      #480,640,3
        #    imshow("original_rgb", ori_bgr)
        #    waitKey()
    if epoch == 0:
        print("\n\nResults saved in {}".format(vis_dir))

    return temp_img, ori_bgr
    #return()


def main():
    if args.dataset == "ycb":
        test_ds = YCB_Dataset('test')
        obj_id = -1
    else:
        test_ds = LM_Dataset('test', cls_type=args.cls)
        obj_id = config.lm_obj_dict[args.cls]
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=20
    )

    rndla_cfg = ConfigRandLA
    trans_cfg = ConfigTrans

    base_best_path = '/home/nachiket/Documents/saved_models/FFB_basic_best/LineMOD/FFB6D_'
    #base_best_path = '/home/nachiket/Documents/train_log_backup/peregrine_models/AttFFB6D_'
    models = {}
    filenames = {}
    for k in config.lm_obj_dict.keys():
        #models[k] = FFB6D(n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg, trans_cfg=trans_cfg, n_kps=config.n_keypoints)
        models[k] = FFB6D(n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg, n_kps=config.n_keypoints)
        filenames[k] = base_best_path+str(k)+'_best.pth.tar'

    #model = FFB6D(    n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg, trans_cfg=trans_cfg, n_kps=config.n_keypoints)
    #model.cuda()

    # load status from checkpoint
    #if args.checkpoint is not None:
    #    load_checkpoint(
    #        model, None, filename=args.checkpoint[:-8]
    #    )

    pcount=0
    for i, data in tqdm.tqdm(
        enumerate(test_loader), leave=False, desc="val"
    ):
        pcount+=1
        if pcount>2:
            break

        predicted_points_list = []
        oriboylist = []
        for k in config.lm_obj_dict.keys():
            obj_id = config.lm_obj_dict[k]
            model = models[k]
            model.cuda()
            load_checkpoint(model, None, filename=filenames[k][:-8])

            new_pts, oriboy = cal_view_pred_pose(model, data, epoch=i, obj_id=obj_id)
            predicted_points_list.append(new_pts)
            oriboylist.append(oriboy)

        ori = oriboylist[0]
        for img in predicted_points_list:
        #    imshow('ppoints: ',img)
        #    waitKey()
            if np.count_nonzero(img) > 0:
                ori = ori + img

        if args.show:
            imshow("projected_pose_rgb", ori)
            imshow("original_rgb", oriboylist[0])
            waitKey()





if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
