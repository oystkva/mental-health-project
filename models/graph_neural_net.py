import os, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROJECT_ROOT,
    ARTIFACT_DIR,
)
from src.data_loader import load_zFCs
from src.functional_connectivity import fisher_z2r

num_node_features = 21**2 ## ?
num_classes = 2

GROUP_TO_LABEL = {
    "HC": 0,
    "MDD": 1,
}


def zFC_to_graph(
    zFC,
    label: int,
    threshold: float = 0.2,
    remove_self_loops: bool = True
):
    """
    Convert zFC matrix into a PyTorch Geometric graph.

    Args:
        zFC: np.ndarray or torch.Tensor, shape [num_nodes, num_nodes]
        label: int, 0 = HC, 1 = MDD
        threshold: keep only edges above this threshold
        remove_self_loops: if True, remove diagonal edges

    Returns:
        torch_geometric.data.Data
    """
    x = torch.as_tensor(zFC, dtype=torch.float32)

    if remove_self_loops:
        x.fill_diagonal_(0.0)
    
    adj = fisher_z2r(x.clone())
    adj = adj.abs()*(adj > threshold)

    edge_index, edge_weight = dense_to_sparse(adj)

    y = torch.as_tensor([label], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
    )

def load_fc_graph_dataset(
    task_type: str,
    band_type: str,
    runs: int,
    atlas_type: str = "Yan2023",
    network_means: bool = True,
    decomp_method: str = "memd",
    groups: tuple[str, ...] = ("HC", "MDD"),
    threshold: float = 0.2,
    remove_self_loops: bool = True,
):
    dataset = []

    for group in groups:
        zFCs = load_zFCs(
            group=group,
            task_type=task_type,
            band_type=band_type,
            runs=runs,
            atlas_type=atlas_type,
            network_means=network_means,
            decomp_method=decomp_method,
            vectorize=False,
        )

        zFCs = np.asarray(zFCs)

        if zFCs.ndim == 2:
            zFCs = zFCs[np.newaxis, ...]

        label = GROUP_TO_LABEL[group]

        for zFC in zFCs:
            graph = zFC_to_graph(
                zFC=zFC,
                label=label,
                threshold=threshold,
                remove_self_loops=remove_self_loops
            )
            dataset.append(graph)

    return dataset


class FCGCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, data):
        x =  data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        batch = data.batch

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x , edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        graph_embedding = global_mean_pool(x, batch)

        logits = self.classifier(graph_embedding)

        return logits
    
    def predict(self, data, return_probabilities = True):
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        
        if return_probabilities:
            return preds, probs
        return preds