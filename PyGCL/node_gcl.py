import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from torch_geometric.utils import subgraph
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
    def __init__(self, encoder, augmentor,train_mask):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.train_mask = train_mask


    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        m1 = global_add_pool(x1, batch)
        m2 = global_add_pool(x2, batch)
        return z, g, x1, x2, g1, g2

def aug(x, edge_index, edge_weight, train_mask, pn):
    x = x.to('cuda')
    train_mask = torch.sigmoid(train_mask)
    subset = torch.where(train_mask < pn, 0, 1)

    num_nodes = edge_index.max().item() + 1
    subset = subset[0:num_nodes]
    subset = subset.nonzero().squeeze()

    edge_index, edge_weight = subgraph(subset, edge_index, edge_weight)
    x = x.clone()
    for i in range(x.shape[0]):
        if i not in edge_index: x[i, :] = 0
    return x, edge_index, edge_weight

class Encoder_mask(torch.nn.Module):
    def __init__(self, train_mask, pn, augmentor):
        super(Encoder_mask, self).__init__()
        self.train_mask = nn.Parameter(train_mask).to('cuda')
        self.augmentor = augmentor
        self.pn = pn

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        # x2, edge_index2, edge_weight2 = aug(x, edge_index, edge_weight1, self.train_mask, self.pn)
        # x2 = x
        # x2 = x2.to('cuda')
        # self.train_mask = torch.sigmoid(self.train_mask)
        # print(self.train_mask)
        # subset = torch.where(self.train_mask < self.pn, 0, 1)
        #
        # num_nodes = edge_index.max().item() + 1
        # subset = subset[0:num_nodes]
        # subset = subset.nonzero().squeeze()
        #
        # edge_index, edge_weight = subgraph(subset, edge_index, edge_weight1)
        # x2 = x2.clone()
        # for i in range(x2.shape[0]):
        #     if i not in edge_index: x2[i, :] = 0


        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        m1 = global_add_pool(x1, batch)
        m2 = global_add_pool(x2, batch)
        return m1, m2

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

def train_m(mask_model, loss_aug_fn, dataloader, optimizer):
    mask_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        m1, m2= mask_model(data.x, data.edge_index, data.batch)
        loss_aug = loss_aug_fn(m1, m2)
        loss_aug = loss_aug.sum(dim=0)/128
        loss_aug.requires_grad_(True)
        loss_aug.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss_aug.item()
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

    #定义train mask
    train_mask = torch.randn((4933,), dtype=torch.float32, requires_grad=True)
    # print(train_mask)

    aug1 = A.Identity()
    aug2 = A.NodeDropping(pn=0.6, train_mask=train_mask)
    # aug2 = A.FeatureMasking(pf=0.1, train_mask=train_mask)
    gconv = GConv(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), train_mask=train_mask).to(device)
    mask_model = Encoder_mask(train_mask=train_mask, pn=0.4, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    #分布训练
    loss_aug_fn = torch.nn.CosineSimilarity(dim = 1)
    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    optimizer_mask = Adam([train_mask], lr=0.01)
    # optimizer_mask = Adam([train_mask], lr=0.01)
    #
    # with tqdm(total=20, desc='(T)') as pbar:
    #     for epoch in range(1, 21):
    #         loss = train_m(mask_model, loss_aug_fn, dataloader, optimizer_mask)
    #         pbar.set_postfix({'loss': loss})
    #         pbar.update()
    #         print(train_mask)
    with tqdm(total=3, desc='(T)') as pbar:
        for epoch in range(1, 3):
            loss = train_m(mask_model, loss_aug_fn, dataloader, optimizer_mask)
            print(loss)
            print(train_mask)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    with tqdm(total=10, desc='(T)') as pbar:
        for epoch in range(1, 11):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    for i in range(3):
        main()
