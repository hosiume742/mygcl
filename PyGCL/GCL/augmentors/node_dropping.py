from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_node
import torch


class NodeDropping(Augmentor):
    def __init__(self, train_mask:torch.Tensor):
        super(NodeDropping, self).__init__()
        self.train_mask = train_mask

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        x, edge_index, edge_weights = drop_node(x, edge_index, edge_weights,  self.train_mask)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
