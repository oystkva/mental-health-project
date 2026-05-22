import os, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from tqdm.auto import tqdm
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
    include_all_runs: bool,
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
            include_all_runs=include_all_runs,
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
            logits = self(data.x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        
        if return_probabilities:
            return preds, probs
        return preds
    
    def fit(
        self, 
        train_loader, 
        val_loader=None, 
        epochs=100, 
        lr=0.001,
        device = None,
        log_every = 10,
    ):
        if device is None:
            device = next(self.parameters()).device

        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_acc_metric = BinaryAccuracy()
        train_f1_metric = BinaryF1Score()

        history = []

        pbar = tqdm(range(epochs), desc="Training")


        for epoch in pbar:
            self.train()
    
            train_acc_metric.reset()
            train_f1_metric.reset()
            
            train_loss_total = 0.0
            n_train = 0

            for batch in train_loader:
                batch = batch.to(device)
                
                optimizer.zero_grad()

                logits = self(batch)
                y = batch.y.view(-1).long()
                
                loss = criterion(logits, batch.y)

                loss.backward()
                optimizer.step()

                batch_size = y.numel()
                train_loss_total += loss.item() * batch_size
                n_train += batch_size

                preds = logits.argmax(dim=1)

                train_acc_metric.update(preds, batch.y)
                train_f1_metric.update(preds, batch.y)
            
            train_loss = train_loss_total / n_train
            train_acc = train_acc_metric.compute().item()
            train_f1 = train_f1_metric.compute().item()

            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
            }
            postfix = {
                "loss": f"{train_loss:.4f}",
                "acc": f"{train_acc:.3f}",
                "f1": f"{train_f1:.3f}",
            }

            if val_loader is not None:
                val_loss, val_acc, val_f1 = self._evaluate_loader(
                    val_loader,
                    criterion=criterion,
                    device=device,
                )

                epoch_result.update(
                    {
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                    }
                )

                postfix.update(
                    {
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc:.3f}",
                        "val_f1": f"{val_f1:.3f}",
                    }
                )

            history.append(epoch_result)

            pbar.set_postfix(postfix)
            
            if (epoch + 1) % log_every == 0:
                tqdm.write(
                    f"Epoch {epoch+1:03d} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Acc: {train_acc:.3f} | "
                    f"F1: {train_f1:.3f}"
                )

        return history

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        if device is None:
            device = next(self.parameters()).device

        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        return self

    def evaluate(self, data, device=None):
        self.eval()
        with torch.no_grad():
            preds = self.predict(data)
            acc = (preds == data.y).float().mean().item()
        print(f'Accuracy: {acc:.4f}')

    def _evaluate_loader(self, loader, criterion=None, device=None):
        if device is None:
            device = next(self.parameters()).device

        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        acc_metric = BinaryAccuracy().to(device)
        f1_metric = BinaryF1Score().to(device)

        self.eval()

        loss_total = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)

                logits = self(batch)
                y = batch.y.view(-1).long()

                loss = criterion(logits, y)

                batch_size = y.numel()
                loss_total += loss.item() * batch_size
                n_samples += batch_size

                preds = logits.argmax(dim=1)

                acc_metric.update(preds, y)
                f1_metric.update(preds, y)

        avg_loss = loss_total / n_samples
        acc = acc_metric.compute().item()
        f1 = f1_metric.compute().item()

        return avg_loss, acc, f1