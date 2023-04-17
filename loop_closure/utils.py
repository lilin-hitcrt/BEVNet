import open3d as o3d
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import torch_scatter
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val**2 * n
        self.var = self.sq_sum / self.count - self.avg**2


def make_open3d_point_cloud(xyz, color=None):
    """construct point cloud from coordinates and colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data):
    """construct features for ransac"""
    feature = o3d.pipelines.registration.Feature()
    feature.resize(data.shape[1], data.shape[0])
    feature.data = data.astype('d').transpose()
    return feature


def load_npy_file(filename):
    return np.load(filename)


def load_npy_files(files):
    out = []
    for file in files:
        out.append(load_npy_file(file))
    return np.array(out)


def load_pcd(filename):
    pc = o3d.io.read_point_cloud(filename)
    return np.asarray(pc.points)


def load_pc_file(filename,
                 coords_range_xyz=[-50., -50, -4, 50, 50, 3],
                 div_n=[256, 256, 32]):
    # pc = o3d.io.read_point_cloud(filename)
    # print(filename)
    pc = np.fromfile(filename, dtype="float32").reshape(-1, 4)[:, :3]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # o3d.visualization.draw_geometries([pcd, ])
    ids, _, _, _ = load_voxel(pc,
                      coords_range_xyz=coords_range_xyz,
                      div_n=div_n)
    voxel_out = np.zeros(div_n, dtype='float32')
    voxel_out[ids[:,0], ids[:,1], ids[:,2]] = 1
    return voxel_out

def load_pc_files(files,
                  coords_range_xyz=[-50., -50, -4, 50, 50, 3],
                  div_n=[256, 256, 32]):
    out = []
    for file in files:
        out.append(load_pc_file(file, coords_range_xyz, div_n))
    return np.array(out, dtype='int32')


def load_voxel(data,
               coords_range_xyz=[-50., -50, -4, 50, 50, 3],
               div_n=[256, 256, 32]):
    div = [(coords_range_xyz[3] - coords_range_xyz[0]) / div_n[0],
           (coords_range_xyz[4] - coords_range_xyz[1]) / div_n[1],
           (coords_range_xyz[5] - coords_range_xyz[2]) / div_n[2]]
    id_x = (data[:, 0] - coords_range_xyz[0]) / div[0]
    id_y = (data[:, 1] - coords_range_xyz[1]) / div[1]
    id_z = (data[:, 2] - coords_range_xyz[2]) / div[2]
    all_id = np.concatenate(
        [id_x.reshape(-1, 1),
         id_y.reshape(-1, 1),
         id_z.reshape(-1, 1)],
        axis=1).astype('int32')
    mask = (all_id[:, 0] >= 0) & (all_id[:, 1] >= 0) & (all_id[:, 2] >= 0) & (
        all_id[:, 0] < div_n[0]) & (all_id[:, 1] < div_n[1]) & (all_id[:, 2] <
                                                                div_n[2])
    all_id = all_id[mask]
    data = data[mask]
    all_id = torch.from_numpy(all_id).long().to(torch.device("cpu"))
    ids, unq_inv, _ = torch.unique(all_id,
                                   return_inverse=True,
                                   return_counts=True,
                                   dim=0)
    ids = ids.detach().cpu().numpy().astype('int32')
    pooled_data = torch_scatter.scatter_mean(torch.from_numpy(data).to(
        torch.device("cpu")),
                                             unq_inv,
                                             dim=0)

    ids_xy, unq_inv_xy, _ = torch.unique(all_id[:, :2],
                                         return_inverse=True,
                                         return_counts=True,
                                         dim=0)
    ids_xy = ids_xy.detach().cpu().numpy().astype('int32')
    pooled_data_xy = torch_scatter.scatter_mean(torch.from_numpy(
        data[:, :2]).to(torch.device("cpu")),
                                                unq_inv_xy,
                                                dim=0)

    # print(pooled_data.shape, data.shape, ids.shape)
    return ids, pooled_data.detach().cpu().numpy(
    ), ids_xy, pooled_data_xy.detach().cpu().numpy()


def se3_to_SE3(se3_pose):
    translate = np.array(se3_pose[0:3], dtype='float32')
    q = se3_pose[3:]
    rot = np.array(R.from_quat(q).as_matrix(), dtype='float32')
    T = np.identity(4, dtype='float32')
    T[:3, :3] = rot
    T[:3, 3] = translate.T
    return T


def read_poses(file, dx=2, st=100):
    stamp = None
    pose = None
    delta_d = dx**2
    with open(file) as f:
        lines = f.readlines()[st:]
    for line in lines:
        line = line.strip().split()
        stampi = line[0]
        posei = [float(line[i]) for i in range(1, len(line))]
        if pose is None:
            pose = [posei]
            stamp = [stampi]
        else:
            diffx = posei[0] - pose[-1][0]
            diffy = posei[1] - pose[-1][1]
            if diffx**2 + diffy**2 > delta_d:
                pose.append(posei)
                stamp.append(stampi)
    pose = np.array(pose, dtype='float32')
    return stamp, pose


def cdist(a, b, metric='euclidean'):
    if metric == 'cosine':
        return torch.sqrt(2 - 2 * torch.matmul(a, b.T))
    elif metric == 'arccosine':
        return torch.acos(torch.matmul(a, b.T))
    else:
        diffs = torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)
        if metric == 'sqeuclidean':
            return torch.sum(diffs**2, dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(torch.sum(diffs**2, dim=-1) + 1e-12)
        elif metric == 'cityblock':
            return torch.sum(torch.abs(diffs), dim=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.
                format(metric))


def rot3d(axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m


def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts


def occ_pcd(points, state_st=6, max_range=np.pi):
    rand_state = random.randint(state_st, 10)
    if rand_state > 9:
        rand_start = random.uniform(-np.pi, np.pi)
        rand_end = random.uniform(rand_start, min(np.pi,
                                                  rand_start + max_range))
        angles = np.arctan2(points[:, 1], points[:, 0])
        return points[(angles < rand_start) | (angles > rand_end)]
    else:
        return points