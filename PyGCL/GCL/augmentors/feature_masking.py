import torch
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature


class FeatureMasking(Augmentor):
    def __init__(self, pf: float, train_mask: torch.Tensor):
        super(FeatureMasking, self).__init__()
        self.pf = pf
        self.train_mask=train_mask

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf, self.train_mask)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
