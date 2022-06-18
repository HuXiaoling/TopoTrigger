import matplotlib

matplotlib.use('Agg')
import time
import torch
import torch.nn as nn
import os
# import visdom
import random
from tqdm import tqdm as tqdm
import sys
# from betti_compute import betti_number
# from TDFMain import *
import numpy
from TDFMain import *


steps = [-1, 1, 100, 150]
scales = [1, 1, 1, 1]
workers = 4
seed = time.time()
step_lr_n_epochs = 10

min_mae = 10000
min_epoch = 0
train_loss_list = []
epoch_list = []
test_error_list = []
epoch_loss = 0

topo_loss = 0
topo_grad = 0

# n = 0;
# topo_cp_map = np.zeros(et_dmap.shape);
n_fix = 0
n_remove = 0
pers_thd_lh = 0.03
pers_thd_gt = 0.03


def getTopoLoss(likelihood):
    # topo_size = likelihood.shape[0]
    topo_size = 20
    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):

            patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                         x:min(x + topo_size, likelihood.shape[1])]
            if (torch.min(patch) == 0 or torch.max(patch) == -1): continue

            pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(patch, 2, 0)
            if (len(pd_lh) == 0): continue

            pd_gt = numpy.array([[0, 1]] * 1)

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0)
            n_fix = 0
            n_remove = 0

            n_fix += len(idx_holes_to_fix)
            n_remove += len(idx_holes_to_remove)
            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                # print('#####################################################################')
                # bcp_lh = bcp_lh + padwidth;
                # dcp_lh = dcp_lh + padwidth;
                for hole_indx in idx_holes_to_fix:

                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                            bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                    # if(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3]):
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                for hole_indx in idx_holes_to_remove:

                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to death  # push to diagonal
                        # if(int(dcp_lh[hole_indx][0]) < likelihood.shape[0] and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1

                            # if(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3]):
                    # if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                    #     0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                    #         likelihood.shape[1]):
                    #     topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                    #         dcp_lh[hole_indx][1])] = 1  # push death to birth # push to diagonal
                    #     # if(int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                    #     # if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                    #     #     0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                    #     #         likelihood.shape[1]):
                    #     #     topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                    #     #         likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                    #     # else:
                    #     #     topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0
                    #
                    #     topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                    #         likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]

    topo_cp_weight_map_tensor = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map_tensor = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()
    loss_topo = (((likelihood * topo_cp_weight_map_tensor) - topo_cp_ref_map_tensor) ** 2).sum()

    # topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    # topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()
    # loss = nn.BCEWithLogitsLoss()
    #
    # loss_topo = loss((likelihood * topo_cp_weight_map), topo_cp_ref_map)
    # print("not scape per: ", inWindows / allWindows, 'loss_topo',loss_topo)

    return loss_topo