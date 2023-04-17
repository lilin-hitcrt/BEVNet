import torch
from matplotlib import pyplot as plt
import spconv.pytorch as spconv
import numpy as np
import os
import random
from scipy.spatial.transform import Rotation as R
import math
import torch_scatter
from copy import deepcopy
import utils


def double_conv_sparse2d(in_channels, out_channels, kernel_size=3):
    return spconv.SparseSequential(
        spconv.SubMConv2d(in_channels, out_channels, kernel_size=kernel_size),
        torch.nn.ReLU(), torch.nn.BatchNorm1d(out_channels),
        spconv.SubMConv2d(out_channels, out_channels, kernel_size=kernel_size),
        torch.nn.ReLU(), torch.nn.BatchNorm1d(out_channels))


class BottleneckSparse2D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super(BottleneckSparse2D, self).__init__()
        self.conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels // 4, 1),
            torch.nn.BatchNorm1d(out_channels // 4), torch.nn.ReLU(),
            spconv.SubMConv2d(out_channels // 4, out_channels // 4,
                              kernel_size),
            torch.nn.BatchNorm1d(out_channels // 4), torch.nn.ReLU(),
            spconv.SubMConv2d(out_channels // 4, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels))
        self.shotcut_conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels))
        self.relu = spconv.SparseSequential(torch.nn.ReLU())

    def forward(self, x):
        y = self.conv(x)
        shortcut = self.shotcut_conv(x)
        y = spconv.functional.sparse_add(y, shortcut)
        return self.relu(y)


class UpBlock(torch.nn.Module):

    def __init__(self, in_channels, shortcut_channels, out_channels) -> None:
        super(UpBlock, self).__init__()
        self.conv1 = BottleneckSparse2D(shortcut_channels, in_channels, 3)
        self.conv2 = BottleneckSparse2D(in_channels, out_channels, 1)

    def forward(self, x, shortcut):
        shortcut = self.conv1(shortcut)
        y = spconv.functional.sparse_add(x, shortcut)
        # y = x.replace_feature(x.features+shortcut.features)
        return self.conv2(y)


class ScoreHead(torch.nn.Module):

    def __init__(self, kernel_size=11) -> None:
        super(ScoreHead, self).__init__()
        self.score_pool = torch.nn.AvgPool2d(kernel_size,
                                             stride=1,
                                             padding=kernel_size // 2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size,
                                           stride=1,
                                           padding=kernel_size // 2)
        self.kernel_size = kernel_size

    def forward(self, x, hard_score):
        fmax = torch.max(x.features) + 1e-6
        x = x.replace_feature(x.features / fmax)
        x_dense = x.dense()

        sum = self.score_pool(x_dense) * self.kernel_size**2
        mask = torch.any(x_dense, dim=1, keepdim=True).float()
        valid_num = self.score_pool(mask) * self.kernel_size**2 + 1e-6
        sum = sum / valid_num
        local_max_score = torch.nn.functional.softplus(x_dense - sum).permute(
            0, 2, 3, 1)
        local_max_score = local_max_score[x.indices[:, 0].long(),
                                          x.indices[:, 1].long(),
                                          x.indices[:, 2].long()]

        depth_wise_max = torch.max(x.features, dim=1,
                                   keepdims=True)[0]  # [n_points, 1]
        depth_wise_max_score = x.features / (1e-6 + depth_wise_max)
        all_scores = local_max_score * depth_wise_max_score
        scores = torch.max(all_scores, dim=1, keepdims=True)[0]
        max_temp = self.max_pool(x_dense).permute(0, 2, 3, 1)
        max_temp = max_temp[x.indices[:, 0].long(), x.indices[:, 1].long(),
                            x.indices[:, 2].long()]
        is_local_max = (x.features == max_temp)
        detected = torch.max(is_local_max.float(), dim=1, keepdims=True)[0]
        scores_hard = scores * detected
        if hard_score:
            scores = scores_hard
        return scores


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1],
                            channels[i],
                            kernel_size=1,
                            bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(torch.nn.InstanceNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class MultiHeadedAttention(torch.nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = torch.nn.ModuleList(
            [deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim,
                                              self.dim * self.num_heads, -1))


class AttentionalPropagation(torch.nn.Module):

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        torch.nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class FeatureFuse(torch.nn.Module):

    def __init__(self, feature_dim, num_heads=1) -> None:
        super(FeatureFuse, self).__init__()
        self.mutihead_attention = AttentionalPropagation(
            feature_dim, num_heads)
        # self.mlp = torch.nn.Conv1d(feature_dim,3,kernel_size=1,bias=True)
    def forward(self, x, source):
        # x=x.permute(1,0).unsqueeze(0)
        # source=source.permute(1,0).unsqueeze(0)
        return (x + self.mutihead_attention(x, source))


class BEVNet(torch.nn.Module):

    def __init__(self, inchannels=64) -> None:
        super(BEVNet, self).__init__()
        self.dconv_down1 = BottleneckSparse2D(inchannels, inchannels * 2, 11)
        self.dconv_down1_1 = BottleneckSparse2D(inchannels * 2, inchannels * 2,
                                                11)
        self.dconv_down2 = BottleneckSparse2D(inchannels * 2, inchannels * 4,
                                              7)
        self.dconv_down2_1 = BottleneckSparse2D(inchannels * 4, inchannels * 4,
                                                7)
        self.dconv_down3 = BottleneckSparse2D(inchannels * 4, inchannels * 8,
                                              5)
        self.dconv_down3_1 = BottleneckSparse2D(inchannels * 8, inchannels * 8,
                                                5)
        self.dconv_down4 = spconv.SubMConv2d(inchannels * 8,
                                             inchannels * 16,
                                             3,
                                             bias=True)
        self.maxpool1 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up1')
        self.maxpool2 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up2')
        self.maxpool3 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up3')
        self.upsample3 = spconv.SparseInverseConv2d(inchannels * 16,
                                                    inchannels * 8,
                                                    kernel_size=3,
                                                    indice_key="up3")
        self.upsample2 = spconv.SparseInverseConv2d(inchannels * 8,
                                                    inchannels * 4,
                                                    kernel_size=3,
                                                    indice_key="up2")
        self.upsample1 = spconv.SparseInverseConv2d(inchannels * 4,
                                                    inchannels * 2,
                                                    kernel_size=3,
                                                    indice_key="up1")
        self.upblock3 = UpBlock(inchannels * 8, inchannels * 8, inchannels * 8)
        self.upblock2 = UpBlock(inchannels * 4, inchannels * 4, inchannels * 4)
        self.upblock1 = UpBlock(inchannels * 2, inchannels * 2, inchannels * 2)
        self.last_conv = spconv.SubMConv2d(inchannels * 2, 32, 1, bias=True)
        self.weight_conv = spconv.SubMConv2d(inchannels * 2,
                                             inchannels,
                                             3,
                                             bias=True)
        self.score_head = ScoreHead(11)
        self.fusenet16 = FeatureFuse(inchannels * 16)
        self.last_conv16 = spconv.SparseSequential(
            spconv.SubMConv2d(inchannels * 16, inchannels * 8, 3, bias=True),
            torch.nn.BatchNorm1d(inchannels * 8), torch.nn.ReLU(),
            spconv.SubMConv2d(inchannels * 8, 1, 3, bias=True))

    def extract_feature(self, x):
        x = spconv.SparseConvTensor.from_dense(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        x = self.dconv_down1_1(x)
        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        x = self.dconv_down2_1(x)
        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        x = self.dconv_down3_1(x)
        x = self.dconv_down4(x)
        return x.dense()

    def calc_overlap(self, x40):
        x40 = spconv.SparseConvTensor.from_dense(x40)
        feature = x40.features
        mask0 = (x40.indices[:, 0] == 0)
        mask1 = (x40.indices[:, 0] == 1)
        fea1 = self.fusenet16(feature[mask0].permute(1, 0).unsqueeze(0),
                              feature[mask1].permute(
                                  1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        fea2 = self.fusenet16(feature[mask1].permute(1, 0).unsqueeze(0),
                              feature[mask0].permute(
                                  1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        x40 = x40.replace_feature(torch.cat([fea1, fea2], dim=0))
        out4 = self.last_conv16(x40)
        out4 = out4.replace_feature(torch.sigmoid(out4.features))
        score0 = out4.features[mask0]
        score1 = out4.features[mask1]

        # im0 = torch.zeros(x40.spatial_shape).float().to(fea1.device)
        # indi0 = x40.indices[mask1].long()
        # im0[indi0[:,1], indi0[:,2]] = score1.reshape(-1)
        # plt.imshow(im0.detach().cpu().numpy())
        # plt.show()

        score_sum0 = torch.sum(score0) / len(score0)
        score_sum1 = torch.sum(score1) / len(score1)
        return (score_sum0 + score_sum1) / 2.

    def forward(self, x, eval=False):
        x = spconv.SparseConvTensor.from_dense(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        x = self.dconv_down1_1(x)
        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        x = self.dconv_down2_1(x)
        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        x = self.dconv_down3_1(x)
        x = self.dconv_down4(x)

        x4 = x
        x40 = x
        feature = x40.features
        mask0 = (x40.indices[:, 0] == 0)
        mask1 = (x40.indices[:, 0] == 1)
        fea1 = self.fusenet16(feature[mask0].permute(1, 0).unsqueeze(0),
                              feature[mask1].permute(
                                  1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        fea2 = self.fusenet16(feature[mask1].permute(1, 0).unsqueeze(0),
                              feature[mask0].permute(
                                  1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        x40 = x40.replace_feature(torch.cat([fea1, fea2], dim=0))
        out4 = self.last_conv16(x40)
        out4 = out4.replace_feature(torch.sigmoid(out4.features))

        x = self.upsample3(x)
        x = self.upblock3(x, conv3)
        x = self.upsample2(x)
        x = self.upblock2(x, conv2)
        x = self.upsample1(x)
        x_last = self.upblock1(x, conv1)
        x = self.last_conv(x_last)
        weight = self.weight_conv(x_last)
        weight_feature = weight.features
        weight_feature = torch.sigmoid(weight_feature)
        weight = weight.replace_feature(weight_feature)
        scores = self.score_head(x, eval)
        features = torch.nn.functional.normalize(x.features, dim=1)
        if eval:
            div = [
                x.spatial_shape[i] // out4.spatial_shape[i]
                for i in range(len(x.spatial_shape))
            ]
            indices = x.indices
            indices[:, 1] = indices[:, 1] // div[0]
            indices[:, 2] = indices[:, 2] // div[1]
            overlap_score = out4.dense().squeeze(1)[indices[:, 0].long(),
                                                    indices[:, 1].long(),
                                                    indices[:, 2].long()]
            scores = scores * overlap_score.unsqueeze(-1)

        x = x.replace_feature(
            torch.cat([features, weight_feature,
                       scores.reshape(-1, 1)], dim=1))
        return x, out4, x4


def gen_random_input(spatial_shape,
                     valid_num=1000,
                     batch_size=1,
                     device=torch.device('cuda')):
    features_out = []
    indices_out = []
    for i in range(batch_size):
        features = torch.rand([valid_num, 1]).float()
        indices_x = torch.randint(0, spatial_shape[0] - 1, [valid_num, 1])
        indices_y = torch.randint(0, spatial_shape[1] - 1, [valid_num, 1])
        indices_z = torch.randint(0, spatial_shape[2] - 1, [valid_num, 1])
        batch_id = torch.ones_like(indices_x) * i
        indices_i = torch.cat([batch_id, indices_x, indices_y, indices_z],
                              dim=1)
        features_out.append(features)
        indices_out.append(indices_i)
    features_out = torch.cat(features_out, dim=0).float().to(device)
    indices_out = torch.cat(indices_out, dim=0).int().to(device)
    input = spconv.SparseConvTensor(features_out, indices_out, spatial_shape,
                                    batch_size)
    return input.dense().squeeze(1)


if __name__ == "__main__":
    device = torch.device("cuda")
    input = gen_random_input([256, 256, 32], 10000, 4)
    net = BEVNet(32)
    net = net.to(device)
    out, out4, x4 = net(input)
    print(out.spatial_shape, out4.spatial_shape, x4.spatial_shape)
