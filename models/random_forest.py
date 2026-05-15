import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
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
from src.data_loader import load_zFC_df, list_networks

BandType = Literal['slow3', 'slow4', 'slow5', 'full', 'all']
TaskType = Literal['restAP', 'restPA', 'all']

from itertools import product
import pandas as pd

import json
import numpy as np
import pandas as pd


import json
import numpy as np
import pandas as pd


def save_cv_report(
    df: pd.DataFrame,
    param_keys: list[str],
    out_path: str,
    primary_metric: str = "f1",
):
    group_cols = [
        "task_type",
        "band_type",
        "param_id",
        *param_keys,
    ]

    metrics = [
        metric for metric in ["accuracy", "precision", "recall", "f1"]
        if metric in df.columns
    ]

    report = {
        "n_folds": int(df["fold"].nunique()),
        "parameter_summary": {},
        "configs": [],
    }

    grouped = df.groupby(group_cols, dropna=False)
    summary_rows = []

    summary_rows = []

    for group_values, group_df in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        config_info = dict(zip(group_cols, group_values))

        group_df = group_df.sort_values("fold")

        depth = config_info.get("max_depth")
        if pd.notna(depth) and float(depth).is_integer():
            depth = int(depth)

        ccp_alpha = config_info.get("ccp_alpha")
        if pd.notna(ccp_alpha) and float(ccp_alpha).is_integer():
            ccp_alpha = int(ccp_alpha)

        config_str = (
            f"[{config_info['task_type']}, "
            f"{config_info['band_type']}, "
            f"n={config_info.get('n_estimators')}, "
            f"depth={depth}, "
            f"min_smpl={config_info.get('min_samples_leaf')}, "
            f"ccp_a={ccp_alpha}]"
        )

        config_entry = {
            "config": config_str,
        }

        summary_row = {
            **config_info,
            "n_folds": int(len(group_df)),
        }

        for metric in metrics:
            vals = group_df[metric].to_numpy(dtype=float)

            vals_list = group_df[metric].round(4).tolist()
            mean_val = round(vals.mean(), 4)
            std_val = round(vals.std(ddof=0), 4)

            config_entry[metric] = (
                f"{vals_list} | mean={mean_val} | std={std_val}"
            )

            summary_row[f"mean_{metric}"] = mean_val
            summary_row[f"std_{metric}"] = std_val

        report["configs"].append(config_entry)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)

    score_col = f"mean_{primary_metric}"

    for param in param_keys:
        comp_df = (
            summary_df
            .groupby(param, dropna=False)[score_col]
            .agg(
                mean="mean",
                std=lambda x: x.std(ddof=0),
                best="max",
            )
            .reset_index()
            .sort_values(param, key=lambda col: col.astype(str))
        )

        values = comp_df[param].tolist()
        means = comp_df["mean"].tolist()
        stds = comp_df["std"].tolist()
        bests = comp_df["best"].tolist()

        value_strs = [
            "None" if pd.isna(v)
            else str(int(v)) if isinstance(v, float) and v.is_integer()
            else f"{v:g}" if isinstance(v, float)
            else str(v)
            for v in values
        ]

        mean_strs = [f"{v:.4g}" for v in means]
        std_strs = [f"{v:.4g}" for v in stds]
        best_strs = [f"{v:.4g}" for v in bests]

        width = max(
            max(len(s) for s in value_strs),
            max(len(s) for s in mean_strs),
            max(len(s) for s in std_strs),
            max(len(s) for s in best_strs),
        )

        report["parameter_summary"][param] = [
            "values: [" + ", ".join(f"{s:>{width}}" for s in value_strs) + "]",
            "mean:   [" + ", ".join(f"{s:>{width}}" for s in mean_strs) + "]",
            "std:    [" + ", ".join(f"{s:>{width}}" for s in std_strs) + "]",
            "best:   [" + ", ".join(f"{s:>{width}}" for s in best_strs) + "]",
        ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=lambda x: x.item())

    return summary_df

def crossval_fc_forest(
    fc_types: dict = {
        'task_types':       ['restAP', 'restPA'],
        'band_types':       ['slow3', 'slow4', 'slow5', 'full'],
    },
    param_grid: dict = {
        'n_estimators':     [10, 40, 70, 100],
        'max_depth':        [5, 10, 15, None],
        'min_samples_leaf': [3, 6, 9],
        'ccp_alpha':        [0.0, 0.03, 0.06],
    },
    folds: int = 5,
    primary_metric: str = 'f1',
    random_state: int = 42,
):
    """
    Cross-validate FCRandomForestClassifier across all task/band combos and pruning params.

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
        X_t, X_test, y_t, y_test = train_test_split(
                                        X, 
                                        y, 
                                        test_size=0.30, 
                                        random_state=random_state,
                                        stratify=y,
                                    )

        scaler = StandardScaler()
        X_t = pd.DataFrame(scaler.fit_transform(X_t), columns=X.columns)
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X.columns)

        best_params, best_score = None, (-1.0, 0.0)
        for param_id, params in enumerate(param_combos):
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_t, y_t), start=1):
                X_train, X_val = X_t.iloc[train_idx], X_t.iloc[val_idx]
                y_train, y_val = y_t.iloc[train_idx], y_t.iloc[val_idx]

                model = FCRandomForestClassifier(
                    band_type=band_type,
                    task_type=task_type,
                    **params
                )
                model.fit(X_train, y_train)
                scores = model.evaluate(X_val, y_val)

                fold_scores.append(scores[primary_metric])

                row = {
                    'param_id': param_id, 
                    'task_type': task_type, 
                    'band_type': band_type, 
                    'fold': fold, **params
                }

                for metric, val in scores.items():
                    row[metric] = val
                
                results.append(row)

            fold_scores = np.asarray(fold_scores)
            mean_score = fold_scores.mean()
            std_score = fold_scores.std()

            if not best_params or mean_score > best_score[0] or (mean_score == best_score[0] and std_score < best_score[1]):
                best_params, best_score = params, (mean_score, std_score)
        

        model = FCRandomForestClassifier(
            task_type=task_type,
            band_type=band_type,
            **best_params,
        )
        model.fit(X_t, y_t)
        model.evaluate(X_test, y_test)
        model.save()

    df = pd.DataFrame(results)
    save_cv_report(df, param_keys, 'cv_report.json')
    df.to_csv('cv_fold_results.csv', index=False)

    return df.sort_values('param_id', ascending=False).reset_index(drop=True)


class FCRandomForestClassifier(RandomForestClassifier):
    def __init__(
        self,
        band_type: BandType,
        task_type: TaskType,
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features='sqrt',
        random_state: int = 42,
        **kwargs
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            **kwargs
        )
        self._band_type = band_type
        self._task_type = task_type
    
    def fit(self, X, y, **kwargs) -> 'FCRandomForestClassifier':
        self.n_samples_fit_ = X.shape[0]
        self.n_features_fit_ = X.shape[1]
        
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
    
    def save(self, save_dir: str = os.path.join(ARTIFACT_DIR, "models", "random_forest")):
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

    @staticmethod
    def _count_stats(values) -> dict:
        values = np.asarray(values, dtype=float)

        return {
            "min": int(values.min()),
            "max": int(values.max()),
            "median": round(float(np.median(values)), 2),
            "mean": round(float(values.mean()), 2),
            "std": round(float(values.std()), 2),
        }
    
    def _check_fitted(self) -> None:
        if not hasattr(self, 'estimators_'):
            raise RuntimeError("Model has not been fitted yet — call fit() first.")

    def _check_evaluated(self) -> None:
        if not hasattr(self, 'eval_scores_'):
            raise RuntimeError("Model has not been evaluated yet — call evaluate() first.")
        
    def _make_stem(self) -> str:
        """Unique filename stem encoding the model's identity."""
        p = self.get_params()
        n_estimators = p['n_estimators']
        max_features = p['max_features']
        depth = p['max_depth']        if p['max_depth']        is not None else "none"
        leaf  = p['min_samples_leaf'] if p['min_samples_leaf'] is not None else 1
        return f"forest_{self.task_type}_{self.band_type}_n{n_estimators}_f{max_features}_d{depth}_l{leaf}"

    def _build_info(self) -> dict:
        has_scores = hasattr(self, "eval_scores_")
    
        depths = [tree.get_depth() for tree in self.estimators_]
        n_leaves = [tree.get_n_leaves() for tree in self.estimators_]
        n_nodes = [tree.tree_.node_count for tree in self.estimators_]

        structure = {
            "n_estimators": int(len(self.estimators_)),
            # "n_features_in": int(self.n_features_in_),

            "depth": self._count_stats(depths),
            "n_leaves": self._count_stats(n_leaves),
            "n_nodes": self._count_stats(n_nodes),

            "split_features": self._build_split_feature_info()
        }

        return {
            "hyperparameters": self.get_params(),
            "fit_data": {
                "n_samples_fit": int(self.n_samples_fit_),
                "n_features_in": int(self.n_features_fit_)
            },
            "structure": structure,
            "eval_scores": (
                {k: round(v, 4) for k, v in self.eval_scores_.items()}
                if has_scores else "not evaluated — call evaluate() first"
            ),
            "saved_at": datetime.now().astimezone().isoformat(timespec='seconds'),
        }

    def _build_split_feature_info(self):
        from collections import Counter
        
        unique_split_features_per_tree = []
        
        split_count = Counter()
        tree_count = Counter()

        for tree in self.estimators_:
            split_features = tree.tree_.feature
            split_features = split_features[split_features >= 0]  # remove leaf nodes
            split_features = [int(f) for f in split_features]

            unique_features = set(split_features)

            unique_split_features_per_tree.append(len(unique_features))

            split_count.update(split_features)
            tree_count.update(unique_features)

        used_feature_indices = sorted(split_count.keys())
        n_used_features = len(used_feature_indices)
    
        feature_labels = make_fc_edge_labels()

        edge_usage = {}
        network_usage = {}

        for f in used_feature_indices:
            label = feature_labels[f]
            edge_usage[label] = f"{split_count[f]} in {tree_count[f]} trees"
            if len(set(label.split(" - "))) == 1:
                network_usage[label.split(" - ")[0]] = network_usage.get(label.split(" - ")[0], 0) + 1    
            else:
                l1, l2 = label.split(" - ")
                network_usage[l1] = network_usage.get(l1, 0) + 1
                network_usage[l2] = network_usage.get(l2, 0) + 1
        info = {
            "unique_features_used_total": n_used_features,
            "unique_features_used_fraction": round(
                n_used_features / self.n_features_in_, 4
            ),
            "unique_features_per_tree": self._count_stats(
                unique_split_features_per_tree
            ),
            "edge_split_usage": edge_usage,
            "network_split_usage": dict(sorted(network_usage.items(), key=lambda item: item[1], reverse=True)),
        },

        return info


def make_fc_edge_labels(include_diagonal=True) -> list[str]:
    """
    Build labels for vectorized upper-triangular FC features.

    The returned list is indexed by feature index:
        edge_labels[feature_idx] -> "Network A - Network B"
    """
    k = 0 if include_diagonal else 1
    network_names = list(list_networks().keys())
    rows, cols = np.triu_indices(len(network_names), k=k)

    edge_labels = [
        f"{network_names[i]} - {network_names[j]}"
        for i, j in zip(rows, cols)
    ]

    return edge_labels

def plot_hyp_param_res_comparison():
    import matplotlib.pyplot as plt


    # compare size of forest
    x = np.ndarray(0)
    y = []
    for file_name in os.listdir(os.path.join(ARTIFACT_DIR, 'models', 'random_forest')):
        if file_name.endswith('.json'):
            with open(os.path.join(ARTIFACT_DIR, 'models', 'random_forest', file_name)) as f:
                data = json.load(f)
                x = np.append(x, data['eval_scores']['f1'])

    noise = np.random.normal(loc=0, size=x.shape)*0.01
    print(*x.shape)
    fig = plt.figure(figsize=(4, 12))
    plt.scatter(np.ones_like(x), x+noise)
    plt.boxplot(x, widths=8)
    print(noise.shape)
    plt.savefig(os.path.join(ARTIFACT_DIR, 'models', 'random_forest', 'box_plot.svg'), format='svg')