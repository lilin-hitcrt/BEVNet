import time
import torch
from net import BEVNet
import numpy as np
from tqdm import tqdm
import open3d as o3d
from matplotlib import pyplot as plt
# from tensorboardX import SummaryWriter
from database import KITTIPairDataset
from sklearn.neighbors import KDTree
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data):
    feature = o3d.pipelines.registration.Feature()
    feature.resize(data.shape[1], data.shape[0])
    feature.data = data.astype('d').transpose()
    return feature


def calc_rte(features_x,
             features_y,
             ind_x,
             ind_y,
             points_xy0,
             points_xy1,
             T_x,
             T_y,
             feature_point_num=1024,
             num_height=64,
             min_z=-4,
             height=7):
    weights_x = features_x[:, 32:32 + num_height]
    weights_y = features_y[:, 32:32 + num_height]
    scores_x = features_x[:, 32 + num_height]
    scores_y = features_y[:, 32 + num_height]
    features_x = features_x[:, :32]
    features_y = features_y[:, :32]
    dz = height / num_height
    pose_x = torch.from_numpy(points_xy0[:, :2]).to(features_x.device).float()
    z_id = torch.arange(0, num_height).float().to(features_x.device).reshape(
        1, num_height) * dz + min_z + dz / 2.
    pos_xz = torch.sum(weights_x * z_id, dim=-1, keepdim=True)
    pose_x = torch.cat([pose_x, pos_xz.view(-1, 1)], dim=1)
    # pose_x[:,2]=0
    score_x = scores_x.detach().cpu().numpy()
    score_order_x = np.argsort(-score_x)
    pose_x1 = pose_x[score_order_x[:feature_point_num]].detach().cpu().numpy()
    best_feature_x1 = features_x[
        score_order_x[:feature_point_num]].detach().cpu().numpy()
    colors = np.zeros_like(pose_x1)
    colors[:, 0] = 255
    pcd_x = make_open3d_point_cloud(pose_x1, colors)
    feat_x = make_open3d_feature(best_feature_x1)

    pose_y = torch.from_numpy(points_xy1[:, :2]).to(features_x.device).float()
    pos_yz = torch.sum(weights_y * z_id, dim=-1, keepdim=True)
    pose_y = torch.cat([pose_y, pos_yz.view(-1, 1)], dim=1)
    # pose_y[:,2]=0
    score_y = scores_y.detach().cpu().numpy()
    score_order_y = np.argsort(-score_y)
    pose_y1 = pose_y[score_order_y[:feature_point_num]].detach().cpu().numpy()
    best_feature_y1 = features_y[
        score_order_y[:feature_point_num]].detach().cpu().numpy()
    colors = np.zeros_like(pose_y1)
    colors[:, 1] = 255
    pcd_y = make_open3d_point_cloud(pose_y1, colors)
    feat_y = make_open3d_feature(best_feature_y1)
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd_x,
        target=pcd_y,
        source_feature=feat_x,
        target_feature=feat_y,
        max_correspondence_distance=0.3,
        mutual_filter=True,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                0.3)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            50000, 1000))
    T_ransac = ransac_result.transformation.astype(np.float32)
    T_diff = T_x @ np.linalg.inv(T_ransac)
    pcd_x.transform(T_ransac)
    # print(T_diff[:3,3])
    rte = np.sqrt(np.sum(T_diff[:3, 3]**2))
    rre = np.arccos((np.trace(T_diff[:3, :3]) - 1) / 2) * 180. / np.pi
    # rre = np.arccos(min(np.trace(T_diff[:3,:3])-1, 2)/2) * 180./np.pi
    # rre = np.arccos((np.trace(T_ransac[:3, :3].transpose() @ T_x[:3, :3]) - 1) / 2)* 180./np.pi
    # print("rte:",rte)
    # o3d.visualization.draw_geometries([pcd_x,pcd_y])
    return rte, rre, np.abs(T_diff[:3, 3])


def inler_ratio(features_x,
                features_y,
                ind_x,
                ind_y,
                points_xy0,
                points_xy1,
                T_x,
                T_y,
                feature_point_num=1024,
                num_height=64,
                min_z=-4,
                height=7):
    weights_x = features_x[:, 32:32 + num_height]
    weights_y = features_y[:, 32:32 + num_height]
    scores_x = features_x[:, 32 + num_height]
    scores_y = features_y[:, 32 + num_height]
    features_x = features_x[:, :32]
    features_y = features_y[:, :32]
    dz = height / num_height
    pose_x = torch.from_numpy(points_xy0[:, :2]).to(features_x.device).float()
    z_id = torch.arange(0, num_height).float().to(features_x.device).reshape(
        1, num_height) * dz + min_z + dz / 2.
    pos_xz = torch.sum(weights_x * z_id, dim=-1, keepdim=True)
    pose_x = torch.cat([pose_x, pos_xz.view(-1, 1)], dim=1)
    score_x = scores_x.detach().cpu().numpy()
    score_order_x = np.argsort(-score_x)
    pose_x1 = pose_x[score_order_x[:feature_point_num]].detach().cpu().numpy()
    best_feature_x = features_x[
        score_order_x[:feature_point_num]].detach().cpu().numpy()
    colors = np.zeros_like(pose_x1)
    colors[:, 0] = 255
    pcd_x = make_open3d_point_cloud(pose_x1, colors)
    pcd_x.transform(T_x)

    pose_y = torch.from_numpy(points_xy1[:, :2]).to(features_x.device).float()
    pos_yz = torch.sum(weights_y * z_id, dim=-1, keepdim=True)
    pose_y = torch.cat([pose_y, pos_yz.view(-1, 1)], dim=1)
    score_y = scores_y.detach().cpu().numpy()
    score_order_y = np.argsort(-score_y)
    pose_y1 = pose_y[score_order_y[:feature_point_num]].detach().cpu().numpy()
    best_feature_y = features_y[
        score_order_y[:feature_point_num]].detach().cpu().numpy()
    colors = np.zeros_like(pose_y1)
    colors[:, 1] = 255
    pcd_y = make_open3d_point_cloud(pose_y1, colors)
    # o3d.visualization.draw_geometries([pcd_x, pcd_y])
    tree = KDTree(best_feature_y)
    ind_nn = tree.query(best_feature_x, 1)[1]
    match_pair = []
    for i in range(len(ind_nn)):
        match_pair.append([i, ind_nn[i][0]])
    match_pair = np.array(match_pair, dtype='int32')
    temp_points1 = np.asarray(pcd_x.points)[match_pair[:, 0]]
    temp_points2 = np.asarray(pcd_y.points)[match_pair[:, 1]]
    diff = np.linalg.norm(temp_points1[:, :3] - temp_points2[:, :3], axis=1)
    diff = diff[diff < 0.3]
    print(len(diff), len(diff) / feature_point_num)
    return len(diff) / feature_point_num


def visualize_corresponds(features_x,
                          features_y,
                          ind_x,
                          ind_y,
                          points_xy0,
                          points_xy1,
                          points_all0,
                          points_all1,
                          T_x,
                          T_y,
                          feature_point_num=1024,
                          num_height=64,
                          min_z=-4,
                          height=7):
    weights_x = features_x[:, 32:32 + num_height]
    weights_y = features_y[:, 32:32 + num_height]
    scores_x = features_x[:, 32 + num_height]
    scores_y = features_y[:, 32 + num_height]
    features_x = features_x[:, :32]
    features_y = features_y[:, :32]
    dz = height / num_height
    pose_x = torch.from_numpy(points_xy0[:, :2]).to(features_x.device).float()
    z_id = torch.arange(0, num_height).float().to(features_x.device).reshape(
        1, num_height) * dz + min_z + dz / 2.
    pos_xz = torch.sum(weights_x * z_id, dim=-1, keepdim=True)
    pose_x = torch.cat([
        pose_x,
        pos_xz.view(-1, 1),
        torch.ones_like(pose_x[:, 0]).view(-1, 1)
    ],
                       dim=1)
    pose_ox = pose_x[:, :3].detach().cpu().numpy()

    pose_x = torch.from_numpy(T_x).float().to(
        device=features_x.device) @ pose_x.T
    pose_x = pose_x.T[:, :3]
    score_x = scores_x.detach().cpu().numpy()

    colors = np.zeros_like(pose_ox)
    colors[:, 0] = 1 * score_x
    pcd = make_open3d_point_cloud(pose_ox, colors)
    o3d.visualization.draw_geometries([
        pcd,
    ])
    # plt.plot(score_x,'.')
    # plt.show()
    score_order_x = np.argsort(-score_x)
    pose_x1 = pose_x[score_order_x[:feature_point_num]].detach().cpu().numpy()
    pose_ox1 = pose_ox[score_order_x[:feature_point_num]]
    best_feature_x = features_x[
        score_order_x[:feature_point_num]].detach().cpu().numpy()
    colors = np.zeros_like(pose_x1)
    colors[:, 0] = 255
    pcd_x = make_open3d_point_cloud(pose_ox, colors)
    pcd_x.transform(T_x)
    pose_y = torch.from_numpy(points_xy1[:, :2]).to(features_x.device).float()
    pos_yz = torch.sum(weights_y * z_id, dim=-1, keepdim=True)
    pose_y = torch.cat([
        pose_y,
        pos_yz.view(-1, 1),
        torch.ones_like(pose_y[:, 0]).view(-1, 1)
    ],
                       dim=1)
    pose_oy = pose_y[:, :3].detach().cpu().numpy()
    pose_y = torch.from_numpy(T_y).float().to(
        device=features_y.device) @ pose_y.T
    pose_y = pose_y.T[:, :3]
    score_y = scores_y.detach().cpu().numpy()
    score_order_y = np.argsort(-score_y)
    pose_y1 = pose_y[score_order_y[:feature_point_num]].detach().cpu().numpy()
    pose_oy1 = pose_oy[score_order_y[:feature_point_num]]
    best_feature_y = features_y[
        score_order_y[:feature_point_num]].detach().cpu().numpy()
    colors = np.zeros_like(pose_y1)
    colors[:, 1] = 255
    pcd_y = make_open3d_point_cloud(pose_oy, colors)
    # o3d.visualization.draw_geometries([pcd_x, pcd_y])

    tree = KDTree(best_feature_y)
    ind_nn = tree.query(best_feature_x, 1)[1]
    match_pair = []
    for i in range(len(ind_nn)):
        match_pair.append([i, ind_nn[i][0]])
    match_pair = np.array(match_pair, dtype='int32')
    temp_points1 = pose_x1[match_pair[:, 0]]
    temp_points2 = pose_y1[match_pair[:, 1]]
    diff = np.linalg.norm(temp_points1 - temp_points2, axis=1)

    temp_pointso1 = pose_ox1[match_pair[:, 0]]
    temp_pointso2 = pose_oy1[match_pair[:, 1]]
    pcd_temp1 = make_open3d_point_cloud(temp_pointso1)
    pcd_temp2 = make_open3d_point_cloud(temp_pointso2)
    pcd_temp1.translate([-50, 0, 0])
    pcd_temp2.translate([50, 0, 0])
    all_points = np.concatenate(
        [np.asarray(pcd_temp1.points),
         np.asarray(pcd_temp2.points)], axis=0)
    match_pair[:, 1] += len(match_pair)
    match_line = o3d.geometry.LineSet(o3d.utility.Vector3dVector(all_points),
                                      o3d.utility.Vector2iVector(match_pair))
    colors = np.zeros([len(match_pair), 3])
    cnum = 0
    diff_mask = (diff < 1)
    colors[diff_mask, 2] = 255
    cnum = np.sum(diff_mask)
    match_line.colors = o3d.utility.Vector3dVector(colors)
    print(cnum, cnum / feature_point_num)

    colors = np.zeros_like(pose_ox)
    colors[score_order_x[:feature_point_num], 0] = 255
    colors[score_order_x[:feature_point_num][diff_mask], 1] = 255
    colors[score_order_x[:feature_point_num][diff_mask], 0] = 0
    pcd_ox = make_open3d_point_cloud(pose_ox, colors)
    # o3d.visualization.draw_geometries([pcd_ox, ])

    match_pair[:, 1] -= len(match_pair)
    colors = np.zeros_like(pose_oy)
    colors[score_order_y[:feature_point_num], 0] = 255
    colors[score_order_y[:feature_point_num][match_pair[:, 1]][diff_mask],
           1] = 255
    colors[score_order_y[:feature_point_num][match_pair[:, 1]][diff_mask],
           0] = 0
    pcd_oy = make_open3d_point_cloud(pose_oy, colors)

    pcd_allx = make_open3d_point_cloud(points_all0, np.zeros_like(points_all0))
    pcd_ally = make_open3d_point_cloud(points_all1, np.zeros_like(points_all1))
    pcd_ox.translate([-50, 0, 0])
    pcd_oy.translate([50, 0, 0])
    pcd_allx.translate([-50, 0, 0])
    pcd_ally.translate([50, 0, 0])
    o3d.visualization.draw_geometries(
        [pcd_ox, pcd_oy, match_line, pcd_allx, pcd_ally])
    # o3d.visualization.draw_geometries([pcd_ox, pcd_oy, pcd_allx, pcd_ally])
    # o3d.visualization.draw_geometries([pcd_ox, pcd_oy])


if __name__ == '__main__':
    states_meter_rte = utils.AverageMeter()
    states_meter_rre = utils.AverageMeter()
    states_meter_recall = utils.AverageMeter()
    states_meter_time = utils.AverageMeter()
    with torch.no_grad():
        feature_point_num = 250
        coords_range_xyz = [-50, -50, -4, 50, 50, 3]
        div = [256, 256, 32]
        net = BEVNet(div[2])
        checkpoint = torch.load('./model/kitti.ckpt')
        net.load_state_dict(checkpoint['state_dict'], False)
        net.to(device=device)
        net.train()
        dataset = KITTIPairDataset([8, 9, 10],
                                   random_rotation=False,
                                   coords_range_xyz=coords_range_xyz,
                                   div=div)

        # for inputi in tqdm(dataset):
        #     input = torch.cat([
        #         torch.from_numpy(inputi['voxel0']).float().unsqueeze(0),
        #         torch.from_numpy(inputi['voxel1']).float().unsqueeze(0)
        #     ],
        #                       dim=0).to(device)
        #     out, _, _ = net(input, True)
        #     mask1 = (out.indices[:, 0] == 0)
        #     mask2 = (out.indices[:, 0] == 1)
        #     visualize_corresponds(out.features[mask1, :],
        #                           out.features[mask2, :],
        #                           out.indices[mask1, :],
        #                           out.indices[mask2, :],
        #                           inputi['points_xy0'],
        #                           inputi['points_xy1'],
        #                           inputi['points0'],
        #                           inputi['points1'],
        #                           inputi['trans0'],
        #                           inputi['trans1'],
        #                           feature_point_num,
        #                           num_height=div[2],
        #                           min_z=coords_range_xyz[2],
        #                           height=coords_range_xyz[5] -
        #                           coords_range_xyz[2])

        all_rte = []
        for inputi in tqdm(dataset):
            input = torch.cat([
                torch.from_numpy(inputi['voxel0']).float().unsqueeze(0),
                torch.from_numpy(inputi['voxel1']).float().unsqueeze(0)
            ],
                              dim=0).to(device)
            torch.cuda.synchronize()
            time0 = time.time()
            out, _, _ = net(input, True)
            torch.cuda.synchronize()
            time1 = time.time()
            states_meter_time.update(time1 - time0)
            mask1 = (out.indices[:, 0] == 0)
            mask2 = (out.indices[:, 0] == 1)
            torch.cuda.synchronize()
            time0 = time.time()
            rtei, rrei, rte_xyz = calc_rte(out.features[mask1, :],
                                           out.features[mask2, :],
                                           out.indices[mask1, :],
                                           out.indices[mask2, :],
                                           inputi['points_xy0'],
                                           inputi['points_xy1'],
                                           inputi['trans0'],
                                           inputi['trans1'],
                                           feature_point_num,
                                           num_height=div[2],
                                           min_z=coords_range_xyz[2],
                                           height=coords_range_xyz[5] -
                                           coords_range_xyz[2])
            torch.cuda.synchronize()
            time1 = time.time()
            all_rte.append(rte_xyz)
            if not np.isnan(rrei) and rtei < 2 and rrei < 5:
                states_meter_rte.update(rtei)
                states_meter_rre.update(rrei)
                states_meter_recall.update(1)
            else:
                states_meter_recall.update(0)
                # print("rte:", rtei, "rre:",rrei)
                # visualize_corresponds(out.features[mask1, :], out.features[mask2, :], out.indices[mask1, :], out.indices[mask2, :],inputi['points_xy0'], inputi['points_xy1'], inputi['trans0'], inputi['trans1'])
            print("RTE", states_meter_rte.avg, states_meter_rte.var)
            print("RRE", states_meter_rre.avg, states_meter_rre.var)
            print("RECALL", states_meter_recall.avg, states_meter_recall.var)
            print("TIME", states_meter_time.avg, states_meter_time.var)
        # all_rte = np.array(all_rte)
        # plt.plot(all_rte[:,0],'b.')
        # plt.plot(all_rte[:,1],'g.')
        # plt.plot(all_rte[:,2],'r.')
        # plt.show()

        # total_inlier = 0
        # num_temp = 0
        # for inputi in tqdm(dataset):
        #     input = torch.cat(
        #             [torch.from_numpy(inputi['voxel0']).float().unsqueeze(0), torch.from_numpy(inputi['voxel1']).float().unsqueeze(0)], dim=0).to(device)
        #     out = net(input)
        #     mask1 = (out.indices[:, 0] == 0)
        #     mask2 = (out.indices[:, 0] == 1)
        #     inlieri = inler_ratio(out.features[mask1, :], out.features[mask2, :], out.indices[mask1, :], out.indices[mask2, :],inputi['points_xy0'],
        #     inputi['points_xy1'], inputi['trans0'], inputi['trans1'],feature_point_num,num_height=div[2], min_z=coords_range_xyz[2],
        #     height=coords_range_xyz[5]-coords_range_xyz[2])
        #     num_temp += 1
        #     total_inlier += inlieri
        #     print('inlier ratio:', total_inlier/num_temp)