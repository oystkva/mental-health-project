import os, sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pydotplus
from typing import Optional, Literal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import pickle as pkl
import json
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROJECT_ROOT,
    ARTIFACT_DIR,
)
from src.data_loader import load_zFC_df

BandType = Literal['slow3', 'slow4', 'slow5', 'full', 'all']
TaskType = Literal['restAP', 'restPA', 'all']

from itertools import product
import pandas as pd

def crossval_fc_trees(
    fc_types: dict = {
        'task_types':       ['restAP', 'restPA'],
        'band_types':       ['slow3', 'slow4', 'slow5', 'full']
    },
    param_grid: dict = {
        'max_depth':        [3, 5, 10, None],
        'min_samples_leaf': [1, 5, 10],
        'ccp_alpha':        [0.0, 0.01, 0.05],
    },
    folds: int = 5,
    random_state: int = 42,
):
    """
    Cross-validate FCDecisionTreeClassifier across all task/band combos and pruning params.

    Args:
        datasets:   {'task_types': [TaskTypeA, TaskTypeB, ...]
                     'band_types': [BandTypeA, BandTypeB, ...]}
        param_grid: pruning hyperparams to search, e.g.:
                    {'max_depth':        [3, 5, None], 
                     'min_samples_leaf': [1, 5, 10],
                     'ccp_alpha':        [0.0, 0.01, 0.05]}
        folds:      number of folds
        random_state: for reproducibility

    Returns:
        pd.DataFrame with mean CV scores per configuration
    """
    # Build all combinations of pruning params
    param_keys = list(param_grid.keys())
    param_combos = [
        dict(zip(param_keys, vals))
        for vals in product(*param_grid.values())
    ]

    results = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    for task_type, band_type in product(fc_types['task_types'], fc_types['band_types']):
        print(f"\n=== Task: {task_type} | Band: {band_type} ===")
        data = load_zFC_df(band_type=band_type, task_type=task_type)
        X, y = data.iloc[:, 0:-1], data.MDD
        X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
        scaler = StandardScaler()
        X_t = pd.DataFrame(scaler.fit_transform(X_t), columns=X.columns)
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X.columns)

        best_params, best_score = None, -np.inf
        for params in param_combos:
            fold_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_t, y_t)):
                X_train, X_val = X_t.iloc[train_idx], X_t.iloc[val_idx]
                y_train, y_val = y_t.iloc[train_idx], y_t.iloc[val_idx]

                if len(np.unique(y_val)) < 2:
                    print(f"Skipping fold {fold} — val set has only one class")
                    continue
                if len(np.unique(y_train)) < 2:
                    print(f"Skipping fold {fold} — train set has only one class")
                    continue

                model = FCDecisionTreeClassifier(
                    band_type=band_type,
                    task_type=task_type,
                    **params
                )
                model.fit(X_train, y_train)
                scores = model.evaluate(X_val, y_val)

                for metric, val in scores.items():
                    fold_scores[metric].append(val)

            # Average across folds
            row = {'task_type': task_type, 'band_type': band_type, **params}
            for metric, vals in fold_scores.items():
                row[f'mean_{metric}'] = round(np.mean(vals), 4)
                row[f'std_{metric}']  = round(np.std(vals), 4)
            if not best_params or row['mean_f1'] > best_score:
                best_params, best_score = params, row['mean_f1']
            results.append(row)

        model = FCDecisionTreeClassifier(
            task_type=task_type,
            band_type=band_type,
            **best_params,
        )
        model.fit(X_t, y_t)
        model.evaluate(X_test, y_test)
        model.save()

    df = pd.DataFrame(results)
    return df.sort_values('mean_f1', ascending=False).reset_index(drop=True)

class FCDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(
        self, 
        band_type: BandType, 
        task_type: TaskType, 
        max_depth: Optional[int] = None,
        min_samples_leaf: float = 1,
        ccp_alpha: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            **kwargs
        )
        self._band_type = band_type
        self._task_type = task_type

    def fit(self, X, y, **kwargs) -> 'FCDecisionTreeClassifier':
        super().fit(X, y, **kwargs)
        return self

    def evaluate(self, X, y):
        self._check_fitted()
        y_pred = self.predict(X)

        self.eval_scores_ = {
            'accuracy':     accuracy_score(y, y_pred),
            'precision':    precision_score(y, y_pred, zero_division=0),
            'recall':       recall_score(y, y_pred, zero_division=0),
            'f1':           f1_score(y, y_pred, zero_division=0)
        }
        return self.eval_scores_
    
    def save(self, save_dir: str = os.path.join(ARTIFACT_DIR, "models", "decision_trees")):
        self._check_fitted()
        os.makedirs(save_dir, exist_ok=True)

        stem = self._make_stem()
        model_path = os.path.join(save_dir, f"{stem}.pkl")
        info_path  = os.path.join(save_dir, f"{stem}.json")

        with open(model_path, 'wb') as f:
            pkl.dump(self, f)
        
        info = self._build_info()
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Saved model → {model_path}")
        print(f"Saved info  → {info_path}")

    def get_eval_scores(self):
        self._check_evaluated()
        s = self.eval_scores_
        return s['accuracy'], s['precision'], s['recall'], s['f1']

    def plot_decision_paths(
        self,
        X,
        n_samples: int = 3,
        out_dir: str = os.path.join(ARTIFACT_DIR, "models", "DT_graph_plots"),
        save_svg: bool = False,
        return_png: bool = False,
    ):
        """
        Visualize the decision tree and highlight the decision paths for a few random samples from X. Saves the visualizations as SVG files in out_dir if save_svg is True, otherwise displays the plots inline.
        Args:
            X (pd.DataFrame): The input features used for the decision tree.
            n_samples (int): Number of random samples from X to visualize the decision paths for.
            out_dir (str): Directory to save the SVG visualizations.
            save_svg (bool): Whether to save the visualizations as SVG files.
        Returns:
            None
        Note:
                - This method requires the graphviz library to be installed and properly configured in the system.
                - The decision paths are highlighted in green, and the number of samples passing through each node is updated accordingly.
        """
        dot_data = self._export_tree_dot(X.columns)
        os.makedirs(out_dir, exist_ok=True)
        # Highlight decision path for one test sample
        sample_idx = np.random.randint(0, len(X), n_samples)
        for k, idx in enumerate(sample_idx):
            graph = pydotplus.graph_from_dot_data(dot_data)

            # Reset all node colors and sample counters
            for node in graph.get_node_list():
                attrs = node.get_attributes()
                if attrs.get("label") is None:
                    continue

                label = attrs["label"]
                if "samples = " in label:
                    parts = label.split("<br/>")
                    for i, part in enumerate(parts):
                        if part.startswith("samples = "):
                            parts[i] = "samples = 0"
                    node.set("label", "<br/>".join(parts))
                    node.set_fillcolor("white")
            sample = X.iloc[[idx]]   # keep as DataFrame
            decision_path = self.decision_path(sample)

            for node_id in decision_path.indices:
                node = graph.get_node(str(node_id))[0]
                node.set_fillcolor("green")

                label = node.get_attributes()["label"]
                parts = label.split("<br/>")
                for i, part in enumerate(parts):
                    if part.startswith("samples = "):
                        current_n = int(part.split("=")[1].strip())
                        parts[i] = f"samples = {current_n + 1}"
                node.set("label", "<br/>".join(parts))

            filename = os.path.join(out_dir, "tree"+str(k)+".svg")
            if save_svg:
                graph.write_svg(filename)
                print(f"Tree visualization saved to {filename}")
            else:
                png_data = graph.create_png()
                bio = BytesIO(png_data)
                img = mpimg.imread(bio)
                if return_png:
                    return img
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                plt.show()

    @property
    def band_type(self) -> str:
        return self._band_type

    @property
    def task_type(self) -> str:
        return self._task_type

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            obj = pkl.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    def _check_fitted(self) -> None:
        if not hasattr(self, 'tree_'):
            raise RuntimeError("Model has not been fitted yet — call fit() first.")

    def _check_evaluated(self) -> None:
        if not hasattr(self, 'eval_scores_'):
            raise RuntimeError("Model has not been evaluated yet — call evaluate() first.")

    def _export_tree_dot(self, feature_names, class_names = ('HC', 'MDD')):
        self._check_fitted()
        return export_graphviz(
            self,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
    
    def _make_stem(self) -> str:
        """Unique filename stem encoding the model's identity."""
        p = self.get_params()
        depth = p['max_depth']        if p['max_depth']        is not None else "none"
        leaf  = p['min_samples_leaf'] if p['min_samples_leaf'] is not None else 1
        alpha = f"{p['ccp_alpha']:.4f}".rstrip('0').rstrip('.')
        return f"tree_{self.task_type}_{self.band_type}_d{depth}_l{leaf}_a{alpha}"

    def _build_info(self) -> dict:
        has_scores = hasattr(self, 'eval_scores_')

        tree = self.tree_
        split_mask = tree.feature >= 0
        split_features = tree.feature[split_mask]
        unique_split_features = np.unique(split_features)

        return {
            "identity": {
                "task_type": str(self.task_type),
                "band_type": str(self.band_type),
            },
            "hyperparameters": self.get_params(),
            "structure": {
                "n_nodes": int(tree.node_count),
                "n_split_nodes": int(split_mask.sum()),
                "n_leaves": int(self.get_n_leaves()),
                "max_depth": int(self.get_depth()),
                "n_features_used": int(len(unique_split_features)),
                "features_used_fraction": round(
                    len(unique_split_features) / self.n_features_in_, 4
                ),
            },
            "eval_scores": (
                {k: round(v, 4) for k, v in self.eval_scores_.items()}
                if has_scores else "not evaluated — call evaluate() first"
            ),
            "saved_at": datetime.now().astimezone().isoformat(timespec='seconds'),
        }