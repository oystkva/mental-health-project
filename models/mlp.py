import json
import os, sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROJECT_ROOT,
    ARTIFACT_DIR,
)
from src.data_loader import load_zFC_df
from src.functional_connectivity import fisher_z2r
from datetime import datetime

# num_features = (434*(434-1))/2 # n upper triangle = n*(n+1)/2 with diagonal - n*(n-1)/2 without
# num_classes = 2

GROUP_TO_LABEL = {
    "HC": 0,
    "MDD": 1,
}


def make_zFC_loader(
    df,
    label_col="MDD",
    batch_size=8,
    shuffle=True,
):
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y = df[label_col].astype(np.int64).to_numpy()

    X = fisher_z2r(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader


class FCMLP(torch.nn.Module):
    def __init__(
            self,
            task: str,
            band: str,
            input_dim,
            hidden_dim: int = 64,
            output_dim: int = 2,
            model_dir = None,
        ):
        super(FCMLP, self).__init__()
        # THese are only to keep track of what data is used if testing diffent fMRI data or frequency bands
        self._task = task
        self._band = band

        self._model_dir = model_dir

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def fit(
        self, 
        train_loader, 
        val_loader=None,
        epochs=100, 
        lr=0.001,
        log_every=10,
    ):
        self._save_model_info()
        
        device = torch.device("cpu")
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

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

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                logits = self(X_batch)
                loss = criterion(logits, y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                n_train += y_batch.size(0)

                preds = logits.argmax(dim=1)
            
                train_acc_metric.update(preds, y_batch)
                train_recall_metric.update(preds, y_batch)
                train_f1_metric.update(preds, y_batch)

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

            if epoch_num % log_every == 0:
                checkpoint_path = os.path.join(
                    str(self.model_dir),
                    f"checkpoint_epoch_{epoch_num:03d}.pt"
                )
                self.save_checkpoint(optimizer=optimizer, path=checkpoint_path, epoch=epoch)

            pbar.set_postfix(self._format_epoch_postfix(epoch_result))
            
            if epoch_num % log_every == 0:
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
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = self.forward(X_batch)
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
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
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
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
            "epoch": epoch,
            "history": history,
            "model_type": "checkpoint",
        }

        torch.save(checkpoint, path)

    @classmethod
    def load_model(cls, path, device=None):
        if device is None:
            device = "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint["config"]

        model = cls(
            task=config["task"],
            band=config["band"],
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            model_dir=os.path.dirname(path),
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
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            model_dir=os.path.dirname(path),
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
    def model_dir(self):
        return self._model_dir

    def _get_device(self):
        return next(self.parameters()).device

    def _generate_model_name(self, base_name: str = "mlp") -> str:
        
        mlp_dir = os.path.join(ARTIFACT_DIR, "models", "mlps")
        os.makedirs(mlp_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = base_name
        
        existing_dirs = [
            d for d in os.listdir(mlp_dir) 
            if os.path.isdir(os.path.join(mlp_dir, d)) and d.startswith(prefix)
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
        model_dir = str(os.path.join(ARTIFACT_DIR, "models", "mlps", self._generate_model_name()))
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    def _ensure_model_dir(self):
        if self._model_dir is None:
            self._model_dir = self._create_model_dir()
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
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = self(X_batch)

                loss = criterion(logits, y_batch)

                batch_size = y_batch.size(0)
                loss_total += loss.item() * batch_size
                n_samples += batch_size

                preds = logits.argmax(dim=1)

                acc_metric.update(preds, y_batch)
                recall_metric.update(preds, y_batch)
                f1_metric.update(preds, y_batch)

        avg_loss = loss_total / n_samples
        acc = acc_metric.compute().item()
        recall = recall_metric.compute().item()
        f1 = f1_metric.compute().item()

        return avg_loss, acc, recall, f1

    def _save_model_info(self):
        model_dir = self._ensure_model_dir()

        info = {"mlp_model_info":
            {
                "task": self.task,
                "band": self.band,
                "atlas": "Schaefer400",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            }
        }
        
        info_path = os.path.join(model_dir, "_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

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