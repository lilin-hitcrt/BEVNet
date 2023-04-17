import torch
from net import BEVNet
from database import KITTIDatasetOverlap, GuangzhouDatasetOverlap
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter
import loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(log_dir,
          coords_range_xyz=[-50, -50, -4, 50, 50, 3],
          div=[256, 256, 32],
          model=""):
    writer = SummaryWriter()
    net = BEVNet(div[2])
    net.to(device=device)
    print(net)
    num_iter = 300050
    train_dataset = KITTIDatasetOverlap(
        # sequs=['07'],
        sequs=[
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
        ],
        root="/media/l/yp2/KITTI/odometry/dataset/sequences/",
        pos_threshold_min=0,
        pos_threshold_max=60,
        neg_thresgold=140,
        coords_range_xyz=coords_range_xyz,
        div_n=div,
        random_rotation=True,
        random_occ=True,
        num_iter=num_iter)
    batch_size = 1
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        net.parameters()),
                                 lr=1e-4,
                                 weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99,
    )
    batch_num = 0
    if not model == "":
        checkpoint = torch.load(model)
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        # start_batch_num = checkpoint['batch_num']
        # optimizer.load_state_dict(checkpoint['optimizer'])
    net.train()
    for i_batch, sample_batch in tqdm(enumerate(train_loader),
                                      total=len(train_loader),
                                      desc='Train',
                                      leave=False):
        optimizer.zero_grad()
        input = torch.cat([sample_batch['voxel0'], sample_batch['voxel1']],
                          dim=0).to(device)
        try:
            out, out4, x4 = net(input)
            mask1 = (out.indices[:, 0] == 0)
            mask2 = (out.indices[:, 0] == 1)
            total_loss, desc_loss, det_loss, score_loss, z_loss, z_loss0, z_loss1, correct_ratio = loss.pair_loss(
                out.features[mask1, :],
                out.features[mask2, :],
                sample_batch['trans0'][0],
                sample_batch['trans1'][0],
                sample_batch['points0'][0],
                sample_batch['points1'][0],
                sample_batch['points_xy0'][0],
                sample_batch['points_xy1'][0],
                num_height=div[2],
                min_z=coords_range_xyz[2],
                height=coords_range_xyz[5] - coords_range_xyz[2],
                search_radiu=max(
                    (coords_range_xyz[3] - coords_range_xyz[0]) / div[0], 0.3))
            loss4, precision4, recall4 = loss.overlap_loss(
                out4, sample_batch['trans0'][0], sample_batch['trans1'][0],
                -50, 50, False)
            desc_loss4, acc4 = loss.dist_loss(x4, sample_batch['trans0'][0],
                                              sample_batch['trans1'][0], -50,
                                              50)
            if not desc_loss4 is None:
                loss_all = desc_loss4 + loss4
            else:
                loss_all = loss4
            if not total_loss is None:
                loss_all = loss_all + total_loss
            loss_all.backward()
            optimizer.step()
            with torch.no_grad():
                if not total_loss is None:
                    writer.add_scalar('desc loss',
                                      desc_loss.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('det loss',
                                      det_loss.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('score loss',
                                      score_loss.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('z loss',
                                      z_loss.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('z loss0',
                                      z_loss0.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('z loss1',
                                      z_loss1.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('correct ratio',
                                      correct_ratio,
                                      global_step=batch_num)
                writer.add_scalar('loss',
                                  loss_all.cpu().item(),
                                  global_step=batch_num)
                writer.add_scalar('loss4',
                                  loss4.cpu().item(),
                                  global_step=batch_num)
                if not desc_loss4 is None:
                    writer.add_scalar('desc loss4',
                                      desc_loss4.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('acc4',
                                      acc4.cpu().item(),
                                      global_step=batch_num)
                writer.add_scalar('precision4',
                                  precision4,
                                  global_step=batch_num)
                writer.add_scalar('recall4', recall4, global_step=batch_num)
                writer.add_scalar(
                    'LR',
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    global_step=batch_num)
                batch_num += 1
                if batch_num % 1000 == 10:
                    torch.save(
                        {
                            'state_dict': net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'batch_num': batch_num
                        }, os.path.join(log_dir,
                                        str(batch_num) + '.ckpt'))
            if batch_num % 1000 == 0 and batch_num:
                scheduler.step()
        except Exception as inst:
            print(inst)


if __name__ == '__main__':
    log_dir = './log_kitti'
    if (not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    train(log_dir,
          coords_range_xyz=[-50, -50, -4, 50, 50, 3],
          div=[256, 256, 32],
          model="./model/kitti.ckpt")
