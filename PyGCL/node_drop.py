import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator, RFEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

import argparse
import logging

#记录器
logger = logging.getLogger('trainlog')
logger.setLevel(logging.DEBUG)

#处理器handler
fileHandler = logging.FileHandler(filename='train_log.txt')
fileHandler.setLevel(logging.DEBUG)

#formatter格式
formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(filename)18s%(lineno)s|%(message)s")

#给处理器设置格式
fileHandler.setFormatter(formatter)

#记录器要设置处理器
logger.addHandler(fileHandler)

parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=128, help='#graphs in batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=64, help='# hidden dim of contrast')
parser.add_argument('--hidden_dim_node', dest='hidden_dim_node', type=int, default=64, help='hidden dim of node_drop')
parser.add_argument('--dataset', dest='dataset', type=str, default='NCI1', help='# dataset name')
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--lr_aug', dest='lr_aug', type=float, default=0.01, help='initial learning rate of aug')
parser.add_argument('--num_layers', dest='num_layers', type=int, default=2, help='num layers of gin')
parser.add_argument('--aug1', dest='aug1', type=str, default='Origin', choices=['Origin', 'Node_Drop', 'Feature_Mask'], help='the first augmentation')
parser.add_argument('--aug2', dest='aug2', type=str, default='Feature_Mask', choices=['Node_Drop', 'Feature_Mask'], help='the second augmentation')

args = parser.parse_args()


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        _, x1 = aug1(x, edge_index, batch)
        _, x2 = aug2(x, edge_index, batch)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index, batch)
        z2, g2 = self.encoder(x2, edge_index, batch)
        return z, g, z1, z2, g1, g2

class Origin(torch.nn.Module):
    def __init__(self):
        super(Origin, self).__init__()

    def forward(self, x, edge_index, batch):
        m = global_add_pool(x, batch)
        return m, x

class Feature_Mask(torch.nn.Module):
    def __init__(self, num_features):
        super(Feature_Mask, self).__init__()
        self.train_mask = nn.Parameter(torch.randn(num_features,), requires_grad=True)

    def forward(self, x, edge_index, batch):
        x = torch.mul(torch.sigmoid(self.train_mask), x)
        m = global_add_pool(x, batch)
        return m, x


class Node_drop(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_node):
        super(Node_drop, self).__init__()
        self.gin1 = make_gin_conv(input_dim, hidden_dim_node)
        self.gin2 = make_gin_conv(hidden_dim_node, 1)

    def forward(self, x, edge_index, batch):
        num_nodes, num_feature = x.size()
        x1 = x
        x2 = x
        x2 = self.gin1(x2, edge_index)
        x2 = F.relu(x2)
        x2 = self.gin2(x2,edge_index)
        x2 = torch.sigmoid(x2)
        x2 = x2.expand(num_nodes, num_feature)
        x2 = x2 * x
        m1 = global_add_pool(x1, batch)
        m2 = global_add_pool(x2, batch)
        return m2, x2

def train_m(aug1, aug2, loss_aug_fn, dataloader, optimizer):
    aug1.train()
    aug2.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        m1, _ = aug1(data.x, data.edge_index, data.batch)
        m2, _ = aug2(data.x, data.edge_index, data.batch)
        loss_aug = loss_aug_fn(m1, m2)
        loss_aug = loss_aug.sum(dim=0)/data.num_graphs
        loss_aug.backward()
        optimizer.step()
        epoch_loss += loss_aug.item()
    return epoch_loss

def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    # result = SVMEvaluator(linear=True)(x, y, split)
    result = RFEvaluator()(x,y,split)
    return result


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    input_dim = max(dataset.num_features, 1)

    if args.aug1 == args.aug2:
        parser.error('aug1 and aug2 should be different!')


    if args.aug1 == 'Origin':
        aug1 = Origin()
    elif args.aug1 == 'Node_Drop':
        aug1 = Node_drop(input_dim=input_dim, hidden_dim_node=args.hidden_dim_node).to(device)
    elif args.aug1 == 'Feature_Mask':
        aug1 = Feature_Mask(input_dim).to(device)

    if args.aug2 == 'Node_Drop':
        aug2 = Node_drop(input_dim=input_dim, hidden_dim_node=args.hidden_dim_node).to(device)
    elif args.aug2 == 'Feature_Mask':
        aug2 = Feature_Mask(input_dim).to(device)

    gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    loss_aug_fn = torch.nn.CosineSimilarity(dim=1)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    lr_aug = args.lr_aug
    lr = args.lr
    logger.debug("DATASET: {}, aug1: {}, aug2: {}, lr_aug: {}, lr: {}".format(args.dataset, args.aug1, args.aug2, lr_aug, lr))
    optimizer_mask = Adam([
        {'params': aug1.parameters()},
        {'params': aug2.parameters()}], lr=lr_aug)
    optimizer = Adam(encoder_model.parameters(), lr=lr)

    epoch = args.epoch

    with tqdm(total=epoch, desc='(T)') as pbar:
        for e in range(1, epoch + 1):
            loss_aug = train_m(aug1, aug2, loss_aug_fn, dataloader, optimizer_mask)
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            logger.debug("aug_epoch: {}, loss_aug: {}, loss {}".format(e, loss_aug, loss))
            pbar.set_postfix({'loss_aug': loss_aug, 'loss':loss})
            pbar.update()

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    logger.debug(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
        main()
