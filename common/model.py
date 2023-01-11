from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
from common.nets.non_local_embedded_gaussian import NONLocalBlock2D
from config import cfg

class ExtendSkeletonGraphConv(nn.Module):
    """
    G-Motion : Multi-hop Mixed graph convolution layer
    """
    def __init__(self, in_features, out_features, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, bias=True):
        super(ExtendSkeletonGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skeleton_graph = cfg.skeleton_graph

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)


        if self.skeleton_graph == 1:
            self.Lamda1 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda2 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
        elif self.skeleton_graph == 2:
            self.Lamda1 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda2 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda3 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
        elif self.skeleton_graph == 3:
            self.Lamda1 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda2 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda3 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda4 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
        elif self.skeleton_graph == 4:
            self.Lamda1 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda2 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda3 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda4 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda5 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        self.adj = adj
        if cfg.NoAffinityModulation is not True:
            self.adj2 = nn.Parameter(torch.ones_like(adj))
            nn.init.constant_(self.adj2, 1e-6)

        self.adj_ext1 = adj_ext1
        if cfg.NoAffinityModulation is not True:
            self.adj_ext1_sub = nn.Parameter(torch.ones_like(adj_ext1))
            nn.init.constant_(self.adj_ext1_sub, 1e-6)

        if self.skeleton_graph == 2 or self.skeleton_graph > 2:
            self.adj_ext2 = adj_ext2
            if cfg.NoAffinityModulation is not True:
                self.adj_ext2_sub = nn.Parameter(torch.ones_like(adj_ext2))
                nn.init.constant_(self.adj_ext2_sub, 1e-6)

        if self.skeleton_graph == 3 or self.skeleton_graph > 3:
            self.adj_ext3 = adj_ext3
            if cfg.NoAffinityModulation is not True:
                self.adj_ext3_sub = nn.Parameter(torch.ones_like(adj_ext3))
                nn.init.constant_(self.adj_ext3_sub, 1e-6)

        if self.skeleton_graph == 4:
            self.adj_ext4 = adj_ext4
            if cfg.NoAffinityModulation is not True:
                self.adj_ext4_sub = nn.Parameter(torch.ones_like(adj_ext4))
                nn.init.constant_(self.adj_ext4_sub, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        if cfg.NoAffinityModulation is not True:
            adj = self.adj.to(input.device) + self.adj2.to(input.device)
        else:
            adj = self.adj.to(input.device)
        adj = (adj.T + adj)/2

        if cfg.NoAffinityModulation is not True:
            adj_ext1 = self.adj_ext1.to(input.device) + self.adj_ext1_sub.to(input.device)
        else:
            adj_ext1 = self.adj_ext1.to(input.device)
        adj_ext1 = (adj_ext1.T + adj_ext1) / 2

        if self.skeleton_graph == 2 or self.skeleton_graph > 2:
            if cfg.NoAffinityModulation is not True:
                adj_ext2 = self.adj_ext2.to(input.device) + self.adj_ext2_sub.to(input.device)
            else:
                adj_ext2 = self.adj_ext2.to(input.device)
            adj_ext2 = (adj_ext2.T + adj_ext2) / 2

        if self.skeleton_graph == 3 or self.skeleton_graph > 3:
            if cfg.NoAffinityModulation is not True:
                adj_ext3 = self.adj_ext3.to(input.device) + self.adj_ext3_sub.to(input.device)
            else:
                adj_ext3 = self.adj_ext3.to(input.device)
            adj_ext3 = (adj_ext3.T + adj_ext3) / 2

        if self.skeleton_graph == 4:
            if cfg.NoAffinityModulation is not True:
                adj_ext4 = self.adj_ext4.to(input.device) + self.adj_ext4_sub.to(input.device)
            else:
                adj_ext4 = self.adj_ext4.to(input.device)
            adj_ext4 = (adj_ext4.T + adj_ext4) / 2

        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        # MM-GCN
        if self.skeleton_graph == 4:
            WHA5 = torch.matmul(adj_ext4 * E, h0) + torch.matmul(adj_ext4 * (1 - E), h1)
            C5 = self.Lamda5 * WHA5
            WHA4 = torch.matmul(adj_ext3 * E, h0) + torch.matmul(adj_ext3 * (1 - E), h1)
            C4 = self.Lamda4 * WHA4
            WHA3 = torch.matmul(adj_ext2 * E, h0) + torch.matmul(adj_ext2 * (1 - E), h1)
            C3 = self.Lamda3 * WHA3
            WHA2 = torch.matmul(adj_ext1 * E, h0) + torch.matmul(adj_ext1 * (1 - E), h1)
            C2 = self.Lamda2 * WHA2
            WHA1 = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
            C1 = self.Lamda1 * WHA1
            output_out = C1 + (1 - self.Lamda1) * C2 + ((1 - self.Lamda1) * ((1 - self.Lamda2) * C3)) + \
                         ((1 - self.Lamda1) * ((1 - self.Lamda2) * ((1 - self.Lamda3) * C4))) + \
                         ((1 - self.Lamda1) * ((1 - self.Lamda2) * ((1 - self.Lamda3) * ((1 - self.Lamda4) * C5))))
        elif self.skeleton_graph == 3:
            WHA4 = torch.matmul(adj_ext3 * E, h0) + torch.matmul(adj_ext3 * (1 - E), h1)
            C4 = self.Lamda4 * WHA4
            WHA3 = torch.matmul(adj_ext2 * E, h0) + torch.matmul(adj_ext2 * (1 - E), h1)
            C3 = self.Lamda3 * WHA3
            WHA2 = torch.matmul(adj_ext1 * E, h0) + torch.matmul(adj_ext1 * (1 - E), h1)
            C2 = self.Lamda2 * WHA2
            WHA1 = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
            C1 = self.Lamda1 * WHA1
            output_out = C1 + (1-self.Lamda1)*C2 + ((1-self.Lamda1) * ((1-self.Lamda2) * C3)) + \
                         ((1-self.Lamda1) * ((1-self.Lamda2) * ((1-self.Lamda3) * C4)))
        elif self.skeleton_graph == 2:
            WHA3 = torch.matmul(adj_ext2 * E, h0) + torch.matmul(adj_ext2 * (1 - E), h1)
            C3 = self.Lamda3 * WHA3
            WHA2 = torch.matmul(adj_ext1 * E, h0) + torch.matmul(adj_ext1 * (1 - E), h1)
            C2 = self.Lamda2 * WHA2
            WHA1 = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
            C1 = self.Lamda1 * WHA1
            output_out = C1 + (1-self.Lamda1)*C2 + ((1-self.Lamda1) * ((1-self.Lamda2) * C3))
        elif self.skeleton_graph == 1:
            WHA2 = torch.matmul(adj_ext1 * E, h0) + torch.matmul(adj_ext1 * (1 - E), h1)
            C2 = self.Lamda2 * WHA2
            WHA1 = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
            C1 = self.Lamda1 * WHA1
            output_out = C1 + (1 - self.Lamda1) * C2

        if self.bias is not None:
            return output_out + self.bias.view(1, 1, -1)
        else:
            return output_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class _GraphConv(nn.Module):
    def __init__(self, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv =  ExtendSkeletonGraphConv(input_dim, output_dim, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class _ResGraphConv(nn.Module):
    def __init__(self, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out

class MMGCN(nn.Module):
    def __init__(self, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, hid_dim, non_local=False, coords_dim=(2, 3), num_layers=4, p_dropout=None):
        super(MMGCN, self).__init__()

        self.non_local = non_local
        _gconv_input = [_GraphConv(adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)

        self.gconv_output = ExtendSkeletonGraphConv(hid_dim, coords_dim[1], adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4)

        if self.non_local:
            self.non_local = NONLocalBlock2D(in_channels=hid_dim, sub_sample=False)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.non_local is False:
            out = self.gconv_input(x)
            out = self.gconv_layers(out)
            out = self.gconv_output(out)
        else:
            out = self.gconv_input(x)
            out = self.gconv_layers(out)
            out = out.unsqueeze(2)
            out = out.permute(0, 3, 2, 1)
            out = self.non_local(out)
            out = out.permute(0, 3, 1, 2)
            out = out.squeeze()
            out = self.gconv_output(out)
        return out

def get_pose_net(cfg, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4):
    model = MMGCN(adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, cfg.channels, non_local=cfg.Non_Local, num_layers=cfg.num_layers, p_dropout=cfg.dropout)
    return model
