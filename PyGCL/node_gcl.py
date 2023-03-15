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
        return z, g, x1, x2, g1, g2, m1, m2

def train(encoder_model, contrast_model, dataloader, optimizer, loss_aug_fn,train_mask):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, x1, x2, g1, g2, m1, m2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss_aug = loss_aug_fn(m1, m2)
        loss_aug2 = loss_aug_fn(x1, x2)
        # print("x1:", x1)
        # print("x2:", x2)
        # print("m1:", m1)
        # print("m2:", m2)
        loss_aug = loss_aug.sum(dim=0) / 128
        print("loss_aug:", loss_aug)
        loss += loss_aug
        train_mask.retain_grad()
        loss_aug.requires_grad_(True)
        loss.backward(retain_graph=True)
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

    #定义train mask
    train_mask = torch.randn((4933,), dtype=torch.float32, requires_grad=True)
    # print(train_mask)

    aug1 = A.Identity()
    aug2 = A.NodeDropping(pn=0.4, train_mask=train_mask)
    # aug2 = A.FeatureMasking(pf=0.1, train_mask=train_mask)
    gconv = GConv(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), train_mask=train_mask).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    #分布训练
    loss_aug_fn = torch.nn.CosineSimilarity(dim = 1)

    optimizer = Adam([{'params': nn.Parameter(train_mask)}, {'params': encoder_model.parameters()}], lr=0.01)
    # optimizer_mask = Adam([train_mask], lr=0.01)
    #
    # with tqdm(total=20, desc='(T)') as pbar:
    #     for epoch in range(1, 21):
    #         loss = train_m(mask_model, loss_aug_fn, dataloader, optimizer_mask)
    #         pbar.set_postfix({'loss': loss})
    #         pbar.update()
    #         print(train_mask)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            print(train_mask)
            loss = train(encoder_model, contrast_model, dataloader, optimizer, loss_aug_fn, train_mask)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    for i in range(3):
        main()
