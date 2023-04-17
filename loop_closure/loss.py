from dis import dis
from soupsieve import match
import torch
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from sklearn.neighbors import KDTree
import random
from sklearn.metrics import precision_recall_fscore_support
import utils


def pair_loss(features_x,
              features_y,
              T_x,
              T_y,
              points0=None,
              points1=None,
              points_xy0=None,
              points_xy1=None,
              num_height=32,
              min_z=-4,
              height=7,
              search_radiu=0.3):
    weights_x = features_x[:, 32:32 + num_height]
    weights_y = features_y[:, 32:32 + num_height]
    scores_x = features_x[:, 32 + num_height]
    scores_y = features_y[:, 32 + num_height]
    features_x = features_x[:, :32]
    features_y = features_y[:, :32]
    points_xy0 = points_xy0.to(device=features_x.device)
    points_xy1 = points_xy1.to(device=features_x.device)
    dz = height / num_height
    pose_x = points_xy0[:, :2].float()
    pose_y = points_xy1[:, :2].float()
    z_id = torch.arange(0, num_height).float().to(features_x.device).reshape(
        1, num_height) * dz + min_z + dz / 2.

    pos_xz = torch.sum(weights_x * z_id, dim=-1, keepdim=True)
    pos_yz = torch.sum(weights_y * z_id, dim=-1, keepdim=True)

    pose_x = torch.cat(
        [pose_x, pos_xz,
         torch.ones_like(pose_x[:, 0]).view(-1, 1)], dim=1)
    pose_y = torch.cat(
        [pose_y, pos_yz,
         torch.ones_like(pose_y[:, 0]).view(-1, 1)], dim=1)
    pose_x = T_x.to(device=features_x.device) @ pose_x.T
    pose_x = pose_x.T[:, :3]
    pose_y = T_y.to(device=features_x.device) @ pose_y.T
    pose_y = pose_y.T[:, :3]
    tempx = pose_x.detach().cpu().numpy()
    tempy = pose_y.detach().cpu().numpy()

    # pcdx = o3d.geometry.PointCloud()
    # pcdx.points=o3d.utility.Vector3dVector(tempx[:,:3])
    # colors = np.zeros([len(pcdx.points),3], dtype='int32')
    # colors[:,0]=255
    # pcdx.colors = o3d.utility.Vector3dVector(colors)
    # pcdy = o3d.geometry.PointCloud()
    # pcdy.points=o3d.utility.Vector3dVector(tempy[:,:3])
    # colors = np.zeros([len(pcdy.points),3], dtype='int32')
    # colors[:,2]=255
    # pcdy.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcdx,pcdy])

    tree = KDTree(tempy[:, :2])
    ind_nn = tree.query_radius(tempx[:, :2], r=search_radiu)
    matchs = []
    for i in range(len(ind_nn)):
        if len(ind_nn[i]) > 0:
            random.shuffle(ind_nn[i])
            matchs.append([i, ind_nn[i][0]])
    if len(matchs) < 1024:
        return None, None, None, None, None, None, None, None
    matchs = np.array(matchs, dtype='int32')
    z_loss = torch.nn.functional.l1_loss(pose_x[matchs[:, 0], 2],
                                         pose_y[matchs[:, 1], 2])
    selected_ind = np.random.choice(range(len(matchs)), 1024, replace=False)
    matchs = matchs[selected_ind]
    score_x = scores_x[matchs[:, 0]]
    score_y = scores_y[matchs[:, 1]]
    match_x = torch.from_numpy(tempx[matchs[:, 0], :2]).to(features_x.device)
    match_y = torch.from_numpy(tempy[matchs[:, 1], :2]).to(features_x.device)
    features_x_selected = features_x[matchs[:, 0]]
    features_y_selected = features_y[matchs[:, 1]]
    desc_loss, acc, _, _, _, dist = circleloss(features_x_selected,
                                               features_y_selected, match_x,
                                               match_y)
    det_loss = detloss(dist, score_x, score_y)
    score_loss = torch.nn.functional.l1_loss(score_x, score_y)

    x_selected = pose_x[matchs[:, 0], :3]
    y_selected = pose_y[matchs[:, 1], :3]
    # z_loss = torch.nn.functional.l1_loss(x_selected[:,2], y_selected[:,2])

    points0 = points0.detach().numpy()
    points1 = points1.detach().numpy()
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(points0)
    pcd0.transform(T_x.detach().numpy())
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.transform(T_y.detach().numpy())
    # o3d.visualization.draw_geometries([pcd0, pcd1, pcdx, pcdy])
    points0 = np.asarray(pcd0.points)
    points1 = np.asarray(pcd1.points)
    tree_x = KDTree(points0)
    ind_nn = tree_x.query(x_selected.detach().cpu().numpy(), 1)[1]
    match_pair = []
    for i in range(len(ind_nn)):
        match_pair.append([i, ind_nn[i][0]])
    match_pair = np.asarray(match_pair, dtype='int32')
    matched_z0 = points0[match_pair[:, 1], 2]
    z_loss0 = torch.nn.functional.l1_loss(
        x_selected[:, 2],
        torch.from_numpy(matched_z0).to(features_x.device).float())

    tree_y = KDTree(points1)
    ind_nn = tree_y.query(y_selected.detach().cpu().numpy(), 1)[1]
    match_pair = []
    for i in range(len(ind_nn)):
        match_pair.append([i, ind_nn[i][0]])
    match_pair = np.asarray(match_pair, dtype='int32')
    matched_z1 = points1[match_pair[:, 1], 2]
    z_loss1 = torch.nn.functional.l1_loss(
        y_selected[:, 2],
        torch.from_numpy(matched_z1).to(features_x.device).float())

    return desc_loss + det_loss + z_loss + z_loss0 + z_loss1, desc_loss, det_loss, score_loss, z_loss, z_loss0, z_loss1, acc


def get_weighted_bce_loss(prediction, gt):
    loss = torch.nn.BCELoss(reduction='none')

    class_loss = loss(prediction, gt)

    weights = torch.ones_like(gt)
    w_negative = gt.sum() / gt.size(0)
    w_positive = 1 - w_negative
    w_negative = max(w_negative, 0.1)
    w_positive = max(w_positive, 0.1)

    weights[gt >= 0.5] = w_positive
    weights[gt < 0.5] = w_negative
    w_class_loss = torch.mean(weights * class_loss)

    #######################################
    # get classification precision and recall
    predicted_labels = prediction.detach().cpu().round().numpy()
    cls_precision, cls_recall, _, _ = precision_recall_fscore_support(
        gt.cpu().numpy(), predicted_labels, average='binary')

    return w_class_loss, cls_precision, cls_recall


def overlap_loss(out, T_x, T_y, min_x=-50, max_x=50, show=False):
    mask0 = (out.indices[:, 0] == 0)
    mask1 = (out.indices[:, 0] == 1)
    indi0 = out.indices[mask0, 1:]
    indi1 = out.indices[mask1, 1:]
    score0 = torch.clamp(out.features[mask0].squeeze(-1), min=0, max=1)
    score1 = torch.clamp(out.features[mask1].squeeze(-1), min=0, max=1)

    dx = (max_x - min_x) / out.spatial_shape[0]
    pose_x = indi0 * dx + min_x + dx / 2.
    pose_y = indi1 * dx + min_x + dx / 2.

    pose_x = torch.cat([
        pose_x,
        torch.zeros_like(pose_x[:, 0]).view(-1, 1),
        torch.ones_like(pose_x[:, 0]).view(-1, 1)
    ],
                       dim=1)
    pose_y = torch.cat([
        pose_y,
        torch.zeros_like(pose_y[:, 0]).view(-1, 1),
        torch.ones_like(pose_y[:, 0]).view(-1, 1)
    ],
                       dim=1)
    pose_x = T_x.to(device=score0.device) @ pose_x.T
    pose_y = T_y.to(device=score0.device) @ pose_y.T
    pose_x = pose_x.T[:, :3]
    pose_y = pose_y.T[:, :3]

    tempx = pose_x.detach().cpu().numpy()
    tempy = pose_y.detach().cpu().numpy()

    tree_y = KDTree(tempy[:, :2])
    ind_nn = tree_y.query(tempx[:, :2], 1)[0].reshape(-1)
    pos_id0 = []
    neg_id0 = []
    for i in range(len(ind_nn)):
        if ind_nn[i] < dx:
            pos_id0.append(i)
        else:
            neg_id0.append(i)

    tree_x = KDTree(tempx[:, :2])
    ind_nn = tree_x.query(tempy[:, :2], 1)[0].reshape(-1)
    pos_id1 = []
    neg_id1 = []
    for i in range(len(ind_nn)):
        if ind_nn[i] < dx:
            pos_id1.append(i)
        else:
            neg_id1.append(i)
    gt0 = torch.zeros_like(score0).to(score0.device)
    gt0[pos_id0] = 1
    gt1 = torch.zeros_like(score1).to(score0.device)
    gt1[pos_id1] = 1
    if show:
        # pcdx = o3d.geometry.PointCloud()
        # pcdx.points=o3d.utility.Vector3dVector(tempx[:,:3])
        # colors = np.zeros([len(pcdx.points),3], dtype='int32')
        # colors[:,0]=255
        # pcdx.colors = o3d.utility.Vector3dVector(colors)
        # pcdy = o3d.geometry.PointCloud()
        # pcdy.points=o3d.utility.Vector3dVector(tempy[:,:3])
        # colors = np.zeros([len(pcdy.points),3], dtype='int32')
        # colors[:,2]=255
        # pcdy.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcdx,pcdy])

        pos_id0 = indi0[pos_id0].long()
        neg_id0 = indi0[neg_id0].long()
        pos_id1 = indi1[pos_id1].long()
        neg_id1 = indi1[neg_id1].long()
        gt_im0 = torch.zeros(out.spatial_shape).to(score0.device)
        gt_im0[pos_id0[:, 0], pos_id0[:, 1]] = 2
        gt_im0[neg_id0[:, 0], neg_id0[:, 1]] = 1
        gt_im1 = torch.zeros(out.spatial_shape).to(score0.device)
        gt_im1[pos_id1[:, 0], pos_id1[:, 1]] = 2
        gt_im1[neg_id1[:, 0], neg_id1[:, 1]] = 1
        pre_im0 = torch.zeros(out.spatial_shape).to(score0.device)
        pre_im0[indi0[:, 0].long(), indi0[:, 1].long()] = score0
        pre_im1 = torch.zeros(out.spatial_shape).to(score0.device)
        pre_im1[indi1[:, 0].long(), indi1[:, 1].long()] = score1
        plt.subplot(2, 2, 1)
        plt.imshow(gt_im0.detach().cpu().numpy())
        plt.subplot(2, 2, 2)
        plt.imshow(pre_im0.detach().cpu().numpy())
        plt.subplot(2, 2, 3)
        plt.imshow(gt_im1.detach().cpu().numpy())
        plt.subplot(2, 2, 4)
        plt.imshow(pre_im1.detach().cpu().numpy())
        plt.show()
    w_class_loss, cls_precision, cls_recall = get_weighted_bce_loss(
        torch.cat([score0, score1]), torch.cat([gt0, gt1]))
    return w_class_loss, cls_precision, cls_recall
    # return torch.nn.functional.binary_cross_entropy_with_logits(score0, gt0) + torch.nn.functional.binary_cross_entropy_with_logits(score1, gt1)


def dist_loss(out, T_x, T_y, min_x=-50, max_x=50):
    mask0 = (out.indices[:, 0] == 0)
    mask1 = (out.indices[:, 0] == 1)
    indi0 = out.indices[mask0, 1:]
    indi1 = out.indices[mask1, 1:]
    fea0 = torch.nn.functional.normalize(out.features[mask0], dim=1)
    fea1 = torch.nn.functional.normalize(out.features[mask1], dim=1)

    dx = (max_x - min_x) / out.spatial_shape[0]
    pose_x = indi0 * dx + min_x + dx / 2.
    pose_y = indi1 * dx + min_x + dx / 2.

    pose_x = torch.cat([
        pose_x,
        torch.zeros_like(pose_x[:, 0]).view(-1, 1),
        torch.ones_like(pose_x[:, 0]).view(-1, 1)
    ],
                       dim=1)
    pose_y = torch.cat([
        pose_y,
        torch.zeros_like(pose_y[:, 0]).view(-1, 1),
        torch.ones_like(pose_y[:, 0]).view(-1, 1)
    ],
                       dim=1)
    pose_x = T_x.to(device=fea0.device) @ pose_x.T
    pose_y = T_y.to(device=fea0.device) @ pose_y.T
    pose_x = pose_x.T[:, :3]
    pose_y = pose_y.T[:, :3]

    tempx = pose_x.detach().cpu().numpy()
    tempy = pose_y.detach().cpu().numpy()

    tree = KDTree(tempy[:, :2])
    ind_nn = tree.query_radius(tempx[:, :2], r=dx)
    matchs = []
    for i in range(len(ind_nn)):
        if len(ind_nn[i]) > 0:
            random.shuffle(ind_nn[i])
            matchs.append([i, ind_nn[i][0]])
    if len(matchs) < 4:
        return None, None
    matchs = np.array(matchs, dtype='int32')
    if len(matchs) > 512:
        selected_ind = np.random.choice(range(len(matchs)), 512, replace=False)
        matchs = matchs[selected_ind]
    features_x_selected = fea0[matchs[:, 0]]
    features_y_selected = fea1[matchs[:, 1]]
    match_x = torch.from_numpy(tempx[matchs[:, 0], :2]).to(fea0.device)
    match_y = torch.from_numpy(tempy[matchs[:, 1], :2]).to(fea0.device)
    desc_loss, acc, _, _, _, _ = circleloss(features_x_selected,
                                            features_y_selected,
                                            match_x,
                                            match_y,
                                            safe_radius=3 * dx,
                                            dist_type='euclidean')
    return desc_loss, acc


def circleloss(anchor,
               positive,
               anchor_keypts,
               positive_keypts,
               dist_type='euclidean',
               log_scale=10,
               safe_radius=1.0,
               pos_margin=0.1,
               neg_margin=1.4):
    dists = utils.cdist(anchor, positive, metric=dist_type)
    dist_keypts = utils.cdist(anchor_keypts,
                              positive_keypts,
                              metric='euclidean')

    pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
    pos_mask = torch.eq(torch.unsqueeze(pids, dim=1),
                        torch.unsqueeze(pids, dim=0))
    neg_mask = dist_keypts > safe_radius

    furthest_positive, _ = torch.max(dists * pos_mask.float(), dim=1)
    closest_negative, _ = torch.min(dists + 1e5 * pos_mask.float(), dim=1)
    average_negative = (torch.sum(dists, dim=-1) -
                        furthest_positive) / (dists.shape[0] - 1)
    diff = furthest_positive - closest_negative
    accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]

    pos = dists - 1e5 * neg_mask.float()
    pos_weight = (pos - pos_margin).detach()
    pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)
    lse_positive_row = torch.logsumexp(log_scale * (pos - pos_margin) *
                                       pos_weight,
                                       dim=-1)
    lse_positive_col = torch.logsumexp(log_scale * (pos - pos_margin) *
                                       pos_weight,
                                       dim=-2)

    neg = dists + 1e5 * (~neg_mask).float()
    neg_weight = (neg_margin - neg).detach()
    neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)
    lse_negative_row = torch.logsumexp(log_scale * (neg_margin - neg) *
                                       neg_weight,
                                       dim=-1)
    lse_negative_col = torch.logsumexp(log_scale * (neg_margin - neg) *
                                       neg_weight,
                                       dim=-2)

    loss_col = torch.nn.functional.softplus(lse_positive_row +
                                            lse_negative_row) / log_scale
    loss_row = torch.nn.functional.softplus(lse_positive_col +
                                            lse_negative_col) / log_scale
    loss = loss_col + loss_row

    return torch.mean(loss), accuracy, furthest_positive.tolist(
    ), average_negative.tolist(), 0, dists


def detloss(dists, anc_score, pos_score):
    pids = torch.FloatTensor(np.arange(len(anc_score))).to(anc_score.device)
    pos_mask = torch.eq(torch.unsqueeze(pids, dim=1),
                        torch.unsqueeze(pids, dim=0))
    furthest_positive, _ = torch.max(dists * pos_mask.float(), dim=1)
    closest_negative, _ = torch.min(dists + 1e5 * pos_mask.float(), dim=1)
    loss = (furthest_positive - closest_negative) * (anc_score +
                                                     pos_score).squeeze(-1)
    return torch.mean(loss)


if __name__ == "__main__":
    pass