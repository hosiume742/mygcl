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
        aug = self.augmentor
        _, _, x1, x2 = aug(x, edge_index, batch)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index, batch)
        z2, g2 = self.encoder(x2, edge_index, batch)
        return z, g, z1, z2, g1, g2


class Node_drop(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Node_drop, self).__init__()
        self.gin1 = make_gin_conv(input_dim, hidden_dim)
        self.gin2 = make_gin_conv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        num_nodes, num_feature = x.size()
        x1 = x
        x2 = x
        x2 = self.gin1(x2, edge_index)
        x2 = F.relu(x2)
        x2 = self.gin2(x2,edge_index)
        x2 = F.sigmoid(x2)
        x2 = x2.expand(num_nodes, num_feature)
        x2 = x2 * x
        m1 = global_add_pool(x1, batch)
        m2 = global_add_pool(x2, batch)
        return m1, m2, x1,x2

def train_m(mask_model, loss_aug_fn, dataloader, optimizer):
    mask_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        m1, m2, _, _ = mask_model(data.x, data.edge_index, data.batch)
        loss_aug = loss_aug_fn(m1, m2)
        loss_aug = loss_aug.sum(dim=0)/128
        loss_aug.requires_grad_(True)
        loss_aug.backward(retain_graph=True)
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
    dataset = TUDataset(path, name='NCI1')
    dataloader = DataLoader(dataset, batch_size=128)
    input_dim = max(dataset.num_features, 1)

    mask_model = Node_drop(input_dim=input_dim, hidden_dim=64).to(device)
    # 使用两种不同的增强方式
    # aug1 = A.Identity()
    # aug2 = A.NodeDropping(train_mask=train_mask)
    gconv = GConv(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor= mask_model).to(device)

    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    loss_aug_fn = torch.nn.CosineSimilarity(dim = 1)
    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    optimizer_mask = Adam(mask_model.parameters(), lr=0.01)
    #

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 10):
            loss_aug = train_m(mask_model, loss_aug_fn, dataloader, optimizer_mask)
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss_aug': loss_aug, 'loss':loss})
            pbar.update()

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
        main()
