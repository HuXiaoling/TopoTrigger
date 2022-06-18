import numpy
import gudhi as gd

def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0, pers_thresh_perfect=0.99, do_return_perfect=False):
    # get persistence list from both diagrams
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_holes = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list();
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list();
    else:

        if (lh_pers.size < gt_n_holes):
            gt_n_holes = lh_pers.size

        tmp = gt_pers > pers_thresh_perfect

        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        print('pers_thresh_perfect', pers_thresh_perfect)
        print('lh_pers > pers_thresh_perfect', (lh_pers > pers_thresh_perfect).sum())
        # print (type(tmp))
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            # if tmp.sum >= 1:
            # n_holes_to_fix = gt_n_holes - lh_n_holes_perfect
            lh_n_holes_perfect = tmp.sum()
            # idx_holes_perfect = np.argpartition(lh_pers, -lh_n_holes_perfect)[
            #                    -lh_n_holes_perfect:]
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
        else:
            # idx_holes_perfect = np.where(lh_pers == lh_pers.max())[0]
            idx_holes_perfect = list();

        # find top gt_n_holes indices
        # idx_holes_to_fix_or_perfect = np.argpartition(lh_pers, -gt_n_holes)[
        #                              -gt_n_holes:]
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        # idx_holes_to_remove = list(
        #    set(range(lh_pers.size)) - set(idx_holes_to_fix_or_perfect))
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    # TODO values below this are small dents so dont fix them; tune this value?
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

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

            patch_flatten = np.asarray(patch).flatten()
            patch_cubic = gd.CubicalComplex(
                dimensions=[topo_size, topo_size],
                top_dimensional_cells=patch_flatten
            )

            patch_per = big_cubic.persistence(homology_coeff_field=2, min_persistence=0)
            patch_pair = big_cubic.cofaces_of_persistence_pairs()

            pd_lh = [patch_per[i][1] for i in range(len(patch_per)) if patch_per[i][0] == 1]

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