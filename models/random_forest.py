import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
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


from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from itertools import product
import os
import json
import numpy as np
import pandas as pd


def crossval_fc_forest(
    fc_types=None,
    param_grid=None,
    folds: int = 5,
    repeats: int = 10,
    primary_metric: str = "f1",
    random_state: int = 42,
):
    if fc_types is None:
        fc_types = {
            "task_types": ["restAP", "restPA"],
            "band_types": ["slow3", "slow4", "slow5", "full"],
        }

    if param_grid is None:
        param_grid = {
            "n_estimators": [200, 250],
            "max_depth": [1, 2],
            "min_samples_leaf": [2, 3],
            "ccp_alpha": [0.0],
        }

    param_keys = list(param_grid.keys())
    param_combos = [
        dict(zip(param_keys, vals))
        for vals in product(*param_grid.values())
    ]

    cv = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=random_state,
    )

    cv_results = []
    selected_results = []

    for task_type, band_type in product(
        fc_types["task_types"],
        fc_types["band_types"],
    ):
        print(f"\n=== Task: {task_type} | Band: {band_type} | run01 only ===")

        data = load_zFC_df(
            band_type=band_type,
            task_type=task_type,
            network_means=True,
            include_all_runs=False,
        )

        data = data.sort_index()

        X = data.iloc[:, 0:-1]
        y = data["MDD"]

        best_params = None
        best_score = (-1.0, np.inf)
        best_param_id = None

        for param_id, params in enumerate(param_combos):
            fold_scores = []

            for split_id, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
                repeat = ((split_id - 1) // folds) + 1
                fold = ((split_id - 1) % folds) + 1

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = FCRandomForestClassifier(
                    band_type=band_type,
                    task_type=task_type,
                    **params,
                )

                model.fit(X_train, y_train)
                scores = model.evaluate(X_val, y_val)

                fold_scores.append(scores[primary_metric])

                row = {
                    "param_id": param_id,
                    "task_type": task_type,
                    "band_type": band_type,
                    "repeat": repeat,
                    "fold": fold,
                    **params,
                }

                for metric, val in scores.items():
                    row[metric] = val

                cv_results.append(row)

            fold_scores = np.asarray(fold_scores, dtype=float)
            mean_score = fold_scores.mean()
            std_score = fold_scores.std(ddof=0)

            if (
                best_params is None
                or mean_score > best_score[0]
                or (
                    mean_score == best_score[0]
                    and std_score < best_score[1]
                )
            ):
                best_params = params
                best_score = (mean_score, std_score)
                best_param_id = param_id

        print(
            f"Best CV {primary_metric}: "
            f"mean={best_score[0]:.4f}, std={best_score[1]:.4f}, "
            f"params={best_params}"
        )

        selected_results.append({
            "task_type": task_type,
            "band_type": band_type,
            "best_param_id": best_param_id,
            "cv_mean_score": best_score[0],
            "cv_std_score": best_score[1],
            **best_params,
        })

        final_model = FCRandomForestClassifier(
            task_type=task_type,
            band_type=band_type,
            **best_params,
        )

        final_model.fit(X, y)
        final_model.save()

    cv_df = pd.DataFrame(cv_results)
    selected_df = pd.DataFrame(selected_results)

    report_dir = os.path.join(ARTIFACT_DIR, "reports", "RF")
    os.makedirs(report_dir, exist_ok=True)

    save_cv_report(
        df=cv_df,
        param_keys=param_keys,
        out_path=os.path.join(report_dir, "cv_report_run01.json"),
        primary_metric=primary_metric,
    )

    cv_df.to_csv(
        os.path.join(report_dir, "cv_fold_results_run01.csv"),
        index=False,
    )

    selected_df.to_csv(
        os.path.join(report_dir, "selected_params_run01.csv"),
        index=False,
    )

    return (
        cv_df
        .sort_values(["task_type", "band_type", "param_id", "repeat", "fold"])
        .reset_index(drop=True),
        selected_df,
    )


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





    
def _make_summary_block(values, means, stds, bests):
    value_strs = [_format_json_value(v) for v in values]
    mean_strs = [f"{v:.4g}" for v in means]
    std_strs = [f"{v:.4g}" for v in stds]
    best_strs = [f"{v:.4g}" for v in bests]

    width = max(
        max(len(s) for s in value_strs),
        max(len(s) for s in mean_strs),
        max(len(s) for s in std_strs),
        max(len(s) for s in best_strs),
    ) + 3

    return [
        "values: [" + ", ".join(f"{s:>{width}}" for s in value_strs) + "]",
        "mean:   [" + ", ".join(f"{s:>{width}}" for s in mean_strs) + "]",
        "std:    [" + ", ".join(f"{s:>{width}}" for s in std_strs) + "]",
        "best:   [" + ", ".join(f"{s:>{width}}" for s in best_strs) + "]",
    ]


def _format_json_value(value):
    if pd.isna(value):
        return "None"

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        value = float(value)

        if value.is_integer():
            return str(int(value))

        return f"{value:g}"

    return str(value)

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

    has_repeats = "repeat" in df.columns

    if has_repeats:
        n_repeats = int(df["repeat"].nunique())
        n_splits = int(df["fold"].nunique())
        n_cv_evaluations = int(
            df[["repeat", "fold"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        n_repeats = 1
        n_splits = int(df["fold"].nunique())
        n_cv_evaluations = n_splits

    report = {
        "n_folds": n_splits,
        "n_repeats": n_repeats,
        "n_cv_evaluations": n_cv_evaluations,
        "primary_metric": primary_metric,
        "parameter_summary": {},
        "condition_summary": {},
        "configs": [],
    }

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    summary_rows = []

    for group_values, group_df in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        config_info = dict(zip(group_cols, group_values))

        sort_cols = ["repeat", "fold"] if has_repeats else ["fold"]
        group_df = group_df.sort_values(sort_cols)

        depth = _format_json_value(config_info.get("max_depth"))
        ccp_alpha = _format_json_value(config_info.get("ccp_alpha"))

        config_str = (
            f"[{config_info['task_type']}, "
            f"{config_info['band_type']}, "
            f"n={_format_json_value(config_info.get('n_estimators'))}, "
            f"depth={depth}, "
            f"min_smpl={_format_json_value(config_info.get('min_samples_leaf'))}, "
            f"ccp_a={ccp_alpha}]"
        )

        config_entry = {
            "config": config_str,
        }

        summary_row = {
            **config_info,
            "n_cv_evaluations": int(len(group_df)),
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
            .groupby(param, dropna=False, sort=False)[score_col]
            .agg(
                mean="mean",
                std=lambda x: x.std(ddof=0),
                best="max",
            )
            .reset_index()
        )

        report["parameter_summary"][param] = _make_summary_block(
            values=comp_df[param].tolist(),
            means=comp_df["mean"].tolist(),
            stds=comp_df["std"].tolist(),
            bests=comp_df["best"].tolist(),
        )

    summary_df["task_band"] = (
        summary_df["task_type"].astype(str)
        + "-"
        + summary_df["band_type"].astype(str)
    )

    for col, name in [
        ("task_type", "task"),
        ("band_type", "band"),
        ("task_band", "task_band"),
    ]:
        comp_df = (
            summary_df
            .groupby(col, dropna=False, sort=False)[score_col]
            .agg(
                mean="mean",
                std=lambda x: x.std(ddof=0),
                best="max",
            )
            .reset_index()
        )

        report["condition_summary"][name] = _make_summary_block(
            values=comp_df[col].tolist(),
            means=comp_df["mean"].tolist(),
            stds=comp_df["std"].tolist(),
            bests=comp_df["best"].tolist(),
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=lambda x: x.item())

    print(json.dumps(report["condition_summary"], indent=4))

    return summary_df