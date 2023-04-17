from net import BEVNet
import torch
import numpy as np
from tqdm import tqdm
import os
import open3d as o3d
import utils
import math
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_feature_kitti(model, files, batch_num=512):
    for q_index in tqdm(range(len(files) // batch_num),
                        total=len(files) // batch_num):
        batch_files = files[q_index * batch_num:(q_index + 1) * batch_num]
        queries = utils.load_pc_files(batch_files)
        with torch.no_grad():
            feed_tensor = torch.tensor(queries).float()
            feed_tensor = feed_tensor.to(next(model.parameters()).device)
            q_out = model.extract_feature(feed_tensor)

            # temp_scores = []
            # for v in q_out:
            #     temp = torch.cat([q_out[0].unsqueeze(0), v.unsqueeze(0)])
            #     score = model.calc_overlap(temp.permute(0, 2, 3 ,1))
            #     temp_scores.append(score.detach().cpu().item())
            # print(temp_scores)
            # plt.plot(temp_scores, '.')
            # plt.show()

            q_out = q_out.detach().cpu().numpy()
            for i in range(len(batch_files)):
                file = batch_files[i].replace('velodyne', "BEV_FEA").replace(
                    '.bin', '.npy')
                np.save(file, q_out[i])

    index_edge = len(files) // batch_num * batch_num
    if index_edge < len(files):
        batch_files = files[index_edge:len(files)]
        queries = utils.load_pc_files(batch_files)
        with torch.no_grad():
            feed_tensor = torch.tensor(queries).float()
            feed_tensor = feed_tensor.to(next(model.parameters()).device)
            q_out = model.extract_feature(feed_tensor).detach().cpu().numpy()
            for i in range(len(batch_files)):
                file = batch_files[i].replace('velodyne', "BEV_FEA").replace(
                    '.bin', '.npy')
                np.save(file, q_out[i])


def extract_kitti(model,
                  seq,
                  batch_num=32,
                  root="/media/l/yp2/KITTI/odometry/dataset/sequences"):
    model.train()
    folder = os.path.join(root, seq, "velodyne")
    out_folder = os.path.join(root, seq, "BEV_FEA")
    if (not os.path.exists(out_folder)):
        os.makedirs(out_folder)
    pcd_files = os.listdir(folder)
    pcd_files.sort()
    pcd_files = [os.path.join(folder, v) for v in pcd_files]
    pose = np.genfromtxt(
        os.path.join(root.replace("sequences", "poses"), seq + ".txt"))
    pose = pose[:, [3, 11]]
    extract_feature_kitti(model, pcd_files, batch_num)


def evaluate_kitti(model,
                   seq="07",
                   batch_num=1,
                   th_min=0,
                   th_max=10,
                   th_max_pre=20,
                   skip=50,
                   root="/media/l/yp2/KITTI/odometry/dataset/sequences"):
    model.train()
    folder = os.path.join(root, seq, 'BEV_FEA')
    feature_files = os.listdir(folder)
    feature_files.sort()
    valid_pose_id = [int(v.split('.')[0]) for v in feature_files]
    feature_files = [os.path.join(folder, v) for v in feature_files]
    pose = np.genfromtxt(
        os.path.join(root.replace("sequences", "poses"), seq + ".txt"))
    pose = pose[:, [3, 11]]
    pose = pose[valid_pose_id]
    pos_num = 0
    neg_num = 0
    for i in tqdm(range(len(feature_files)), total=len(feature_files)):
        diff = pose - pose[i]
        dis = np.linalg.norm(diff, axis=1)
        max_d = 100000
        # dis[max(i-skip,0):min(i+skip,len(dis))]=max_d
        dis[max(i - skip, 0):] = max_d
        mask = (dis < th_min)
        dis[mask] = max_d
        minid_gt = np.argmin(dis)
        temp_min = np.min(dis)
        if temp_min < th_max:
            scores = np.zeros(len(feature_files), dtype='float32')
            feai = utils.load_npy_files([feature_files[i]])
            feai = feai.repeat(batch_num, axis=0)
            feai = torch.from_numpy(feai).to(next(model.parameters()).device)
            st = 0
            while st < min(len(feature_files), i - skip):
                ed = min(st + batch_num, len(feature_files))
                batch_files = feature_files[st:ed]
                feai = feai[0:len(batch_files)]
                feaj = utils.load_npy_files(batch_files)
                feaj = torch.from_numpy(feaj).to(
                    next(model.parameters()).device)
                overlap = model.calc_overlap(
                    torch.cat([feai, feaj]).permute(0, 2, 3, 1))
                scores[st:ed] = overlap.detach().cpu().numpy()
                st = ed
            scores[max(i - skip, 0):] = 0
            scores[mask] = 0
            minid = np.argmax(scores)
            mindis = math.sqrt((pose[i, 0] - pose[minid, 0])**2 +
                               (pose[i, 1] - pose[minid, 1])**2)
            if mindis < th_max_pre:
                pos_num += 1
            else:
                neg_num += 1
            print("recall:", pos_num / (pos_num + neg_num), mindis, temp_min,
                  pos_num, neg_num + pos_num, i, minid, minid_gt)


if __name__ == "__main__":
    net = BEVNet(32).to(device=device)
    checkpoint = torch.load('./model/kitti.ckpt')
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    net.train()
    extract_kitti(net, "07", 64)
    evaluate_kitti(net,
                   seq="07",
                   batch_num=1,
                   th_min=0,
                   th_max=5,
                   th_max_pre=10,
                   skip=50)
