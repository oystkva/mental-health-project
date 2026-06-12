import json
import os, sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall
from tqdm.auto import tqdm
# from torch_geometric.loader import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROJECT_ROOT,
    ARTIFACT_DIR,
)
from src.data_loader import load_zFCs
from src.functional_connectivity import fisher_z2r
from datetime import datetime


# num_node_features = 434 # edges in matrix
# num_classes = 2

GROUP_TO_LABEL = {
    "HC": 0,
    "MDD": 1,
}


class CrossEntropyRecallPenaltyLoss(nn.Module):
    def __init__(
        self,
        recall_weight: float = 0.25,
        positive_class: int = 1,
        class_weights=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.recall_weight = recall_weight
        self.positive_class = positive_class
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.view(-1).long()

        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)[:, self.positive_class]
        targets_pos = (targets == self.positive_class).float()

        n_pos = targets_pos.sum()

        if n_pos == 0:
            recall_penalty = torch.tensor(0.0, device=logits.device)
        else:
            recall_penalty = ((1.0 - probs) * targets_pos).sum() / (n_pos + self.eps)

        return ce_loss + self.recall_weight * recall_penalty

class SoftFBetaLoss(nn.Module):
    def __init__(self, beta=2.0, positive_class=1, eps=1e-8):
        super().__init__()
        self.beta = beta
        self.positive_class = positive_class
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.view(-1)
        
        probs = F.softmax(logits, dim=1)[:, self.positive_class]
        targets = (targets == self.positive_class).float()

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        beta2 = self.beta ** 2

        soft_fbeta = ((1 + beta2) * tp + self.eps) / (
            (1 + beta2) * tp + beta2 * fn + fp + self.eps
        )

        return 1 - soft_fbeta

class CompoundProbabilityRecallLoss(nn.Module):
    """
    Compound loss combining a base probability/classification loss
    with a soft F-beta loss.

    Useful when you want:
        - meaningful prob_MDD values
        - stronger emphasis on MDD recall
    """

    def __init__(
        self,
        beta: float = 2.0,
        fbeta_weight: float = 0.25,
        positive_class: int = 1,
        class_weights=None,
    ):
        super().__init__()

    
        self.base_loss = nn.CrossEntropyLoss(weight=class_weights)

        self.fbeta_loss = SoftFBetaLoss(
            beta=beta,
            positive_class=positive_class,
        )

        self.fbeta_weight = fbeta_weight

    def forward(self, logits, targets):
        base = self.base_loss(logits, targets)
        fbeta = self.fbeta_loss(logits, targets)

        return base + self.fbeta_weight * fbeta


def zFC_to_graph(
    zFC,
    label: int,
    threshold: float = 0.2,
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
    x = torch.as_tensor(zFC, dtype=torch.float32).clone()
    
    adj = fisher_z2r(x.clone())
    adj = adj*(adj.abs() > threshold)

    edge_index, edge_weight = dense_to_sparse(adj.abs())

    y = torch.as_tensor([label], dtype=torch.long)

    return Data(
        x=fisher_z2r(x),
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
    )

def load_fc_graph_dataset(
    task_type: str,
    band_type: str,
    include_all_runs: bool,
    atlas_type: str = "Schaefer400",
    network_means: bool = True,
    decomp_method: str = "memd",
    groups: tuple[str, ...] = ("HC", "MDD"),
    threshold: float = 0.5,
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
            )
            dataset.append(graph)

    return dataset


class FCGCN(torch.nn.Module):
    def __init__(
        self,
        task: str,
        band: str,
        atlas: str,
        num_node_features: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        model_dir=None,
        dropout: float = 0.3,
    ):
        super().__init__()
        # THese are only to keep track of what data is used if testing diffent fMRI data or frequency bands
        self._task = task
        self._band = band
        self._atlas = atlas

        self._model_dir = model_dir

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.classifier = Linear(hidden_channels, num_classes)

        self.dropout = dropout

    def forward(self, data):
        x =  data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        batch = data.batch

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x , edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        graph_embedding = global_mean_pool(x, batch)

        logits = self.classifier(graph_embedding)

        return logits
    
    def fit(
        self, 
        train_loader, 
        val_loader=None, 
        epochs=100, 
        label_smoothing=0.0, 
        lr=0.001,
        log_every=10,
        weight_decay=1e-4,
        device=None
    ):
        self._ensure_model_dir()
        if device is None:        
            device = torch.device("cpu")
        else:
            device = torch.device(device)
        self.to(device)


        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        class_weights = torch.tensor([1.0, 1.7], device=device)

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)

        self._save_train_info(
            criterion=type(criterion).__name__,
            optimizer=type(optimizer).__name__,
            label_smoothing=label_smoothing,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=train_loader.batch_size,
        )

        train_acc_metric = BinaryAccuracy().to(device)
        train_recall_metric = BinaryRecall().to(device)
        train_f1_metric = BinaryF1Score().to(device)

        history = []

        pbar = tqdm(range(epochs), desc="Training")

        for epoch in pbar:
            self.train()
    
            train_acc_metric.reset()
            train_recall_metric.reset()
            train_f1_metric.reset()
            
            train_loss = 0.0
            n_train = 0


            for batch in train_loader:
                batch = batch.to(device)
                
                optimizer.zero_grad()

                logits = self(batch)
                y = batch.y.view(-1).long()
                
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                batch_size = y.numel()
                train_loss += loss.item() * batch_size
                n_train += batch_size

                preds = logits.argmax(dim=1)

                train_acc_metric.update(preds, y)
                train_recall_metric.update(preds, y)
                train_f1_metric.update(preds, y)
            
            train_loss /= n_train
            train_acc = train_acc_metric.compute().item()
            train_recall = train_recall_metric.compute().item()
            train_f1 = train_f1_metric.compute().item()

            #region epoch book-keeping, validation, tqdm progress, and checkpoints
            epoch_num = epoch + 1

            epoch_result = {
                "epoch": epoch_num,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_recall": train_recall,
                "train_f1": train_f1,
            }

            if val_loader is not None:
                val_loss, val_acc, val_recall, val_f1 = self._evaluate_loader(
                    val_loader,
                    criterion=criterion,
                )

                epoch_result.update(
                    {
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_recall": val_recall,
                        "val_f1": val_f1,
                    }
                )

            if (epoch_num) % log_every == 0:
                checkpoint_path = os.path.join(
                    str(self.model_dir),
                    f"checkpoint_epoch_{epoch_num:03d}.pt"
                )
                self.save_checkpoint(optimizer=optimizer, path=checkpoint_path, epoch=epoch_num, history=history)

            pbar.set_postfix(self._format_epoch_postfix(epoch_result))
            
            if (epoch_num) % log_every == 0:
                self._print_epoch_result(epoch_result)

            history.append(epoch_result)
            #endregion
        self.save(epoch=epochs)
        return history

    def predict(self, loader, return_probabilities=True):
        device = self._get_device()

        self.eval()

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)

                logits = self(batch)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

        preds = torch.cat(all_preds, dim=0)
        probs = torch.cat(all_probs, dim=0)

        if return_probabilities:
            return preds, probs
        return preds    

    def evaluate(self, loader):
        loss, acc, recall, f1 = self._evaluate_loader(loader)

        print(
            f"Loss: {loss:.4f} | "
            f"Accuracy: {acc:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        return {
            "loss": loss,
            "acc": acc,
            "recall": recall,
            "f1": f1,
        }

    def save(self, path=None, epoch=None):
        model_dir = self._ensure_model_dir()

        if path is None:
            path = os.path.join(model_dir, "model_final.pt")

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "task": self.task,
                "band": self.band,
                "atlas": self.atlas,
                "num_node_features": self.num_node_features,
                "hidden_channels": self.hidden_channels,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
            },
            "epoch": epoch,
            "model_type": "final",
        }

        torch.save(checkpoint, path)
    
    def save_checkpoint(self, optimizer, path=None, epoch=None, history=None):
        model_dir = self._ensure_model_dir()

        if path is None:
            if epoch is None:
                path = os.path.join(model_dir, "checkpoint.pt")
            else:
                path = os.path.join(model_dir, f"checkpoint_epoch_{epoch:03d}.pt")

        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "task": self.task,
                "band": self.band,
                "atlas": self.atlas,
                "num_node_features": self.num_node_features,
                "hidden_channels": self.hidden_channels,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
            },
            "epoch": epoch,
            "history": history,
            "model_type": "checkpoint",
        }

        torch.save(checkpoint, path)
    
    def get_subject_prediction_scores(self, loader):
        """
        Return a compact per-sample prediction summary.

        Assumes:
            0 = HC
            1 = MDD
        """
        device = self._get_device()
        self.eval()

        label_names = {
            0: "HC",
            1: "MDD",
        }

        rows = []
        subject_number = 1

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)

                logits = self(batch)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                y = batch.y.view(-1).long()

                for i in range(y.numel()):
                    true_label = int(y[i].cpu())
                    pred_label = int(preds[i].cpu())

                    rows.append({
                        "subject": subject_number,
                        "true": label_names[true_label],
                        "pred": label_names[pred_label],
                        "prob_MDD": float(probs[i, 1].cpu()),
                        "correct": pred_label == true_label,
                    })

                    subject_number += 1

        return pd.DataFrame(rows)

    @classmethod
    def load_model(cls, path, device=None):
        if device is None:
            device = "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint["config"]

        model = cls(
            task=config["task"],
            band=config["band"],
            atlas=config["atlas"],
            num_node_features=config["num_node_features"],
            hidden_channels=config["hidden_channels"],
            num_classes=config["num_classes"],
            model_dir=os.path.dirname(path),
            dropout=config.get("dropout", 0.3),
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        return model
    
    @classmethod
    def load_checkpoint(cls, path, lr=0.001, device=None):
        if device is None:
            device = "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]

        model = cls(
            task=config["task"],
            band=config["band"],
            atlas=config["atlas"],
            num_node_features=config["num_node_features"],
            hidden_channels=config["hidden_channels"],
            num_classes=config["num_classes"],
            model_dir=os.path.dirname(path),
            dropout=config.get("dropout", 0.3),
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", [])

        return model, optimizer, start_epoch, history

    @property
    def band(self) -> str:
        return self._band

    @property
    def task(self) -> str:
        return self._task
    
    @property
    def atlas(self) -> str:
        return self._atlas
    
    @property
    def model_dir(self):
        return self._model_dir

    def _get_device(self):
        return next(self.parameters()).device

    def _save_model_info(self):
        info = {"gcn_model_info":
            {
                "task": self.task,
                "band": self.band,
                "atlas": self.atlas,
                "num_node_features": self.num_node_features,
                "hidden_channels": self.hidden_channels,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
            }
        }
        
        info_path = os.path.join(str(self._model_dir), "_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def _save_train_info(
        self,
        criterion: str,
        optimizer: str,
        lr: float,
        label_smoothing: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
    ):
        model_dir = self._ensure_model_dir()
        info_path = os.path.join(model_dir, "_info.json")

        with open(info_path, 'r') as f:
            info = json.load(f)

        info["train_info"] = {
            "criterion": criterion,
            "label_smoothing": label_smoothing,
            "optimizer": optimizer,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
        }

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    def _generate_model_name(self, base_name: str = "gcn") -> str:
        
        gnn_dir = os.path.join(ARTIFACT_DIR, "models", "gcns")
        os.makedirs(gnn_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = base_name
        
        existing_dirs = [
            d for d in os.listdir(gnn_dir) 
            if os.path.isdir(os.path.join(gnn_dir, d)) and d.startswith(prefix)
        ]
        
        model_id = "001"
        if existing_dirs:
            model_id = max(
                [d.split("_")[1] for d in existing_dirs if len(d.split("_")) > 1]
                + [model_id]            )
            model_id = str(int(model_id) + 1).zfill(3)
        
        model_name = f"{prefix}_{model_id}_{timestamp}"
        return model_name

    def _create_model_dir(self):
        model_dir = str(os.path.join(ARTIFACT_DIR, "models", "gcns", self._generate_model_name()))
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _ensure_model_dir(self):
        if self._model_dir is None:
            self._model_dir = self._create_model_dir()
            self._save_model_info()
        return self._model_dir

    def _evaluate_loader(self, loader, criterion=None):
        device = self._get_device()

        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        acc_metric = BinaryAccuracy().to(device)
        recall_metric = BinaryRecall().to(device)
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
                recall_metric.update(preds, y)
                f1_metric.update(preds, y)

        avg_loss = loss_total / n_samples
        acc = acc_metric.compute().item()
        recall = recall_metric.compute().item()
        f1 = f1_metric.compute().item()

        return avg_loss, acc, recall, f1

    def _format_epoch_postfix(self, epoch_result):
        display_names = {
            "train_loss": "loss",
            "train_acc": "acc",
            "train_recall": "recall",
            "train_f1": "f1",
            "val_loss": "val_loss",
            "val_acc": "val_acc",
            "val_recall": "val_recall",
            "val_f1": "val_f1",
        }

        return {
            display_names[key]: f"{epoch_result[key]:.4f}"
            for key in display_names
            if key in epoch_result
        }
    
    def _print_epoch_result(self, epoch_result):
        keys = [
            "train_loss",
            "train_acc",
            "train_recall",
            "train_f1",
            "val_loss",
            "val_acc",
            "val_recall",
            "val_f1",
        ]

        parts = [f"Epoch {epoch_result['epoch']:03d}"]

        for key in keys:
            if key in epoch_result:
                parts.append(f"{key}: {epoch_result[key]:.4f}")

        tqdm.write(" | ".join(parts))