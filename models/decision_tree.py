import os, sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
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

from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold
import os
import numpy as np
import pandas as pd


def crossval_fc_trees(
    fc_types=None,
    param_grid=None,
    folds: int = 5,
    repeats: int = 10,
    primary_metric: str = "f1",
    random_state: int = 42,
):
    """
    Run run01-only repeated cross-validation for Decision Tree classification.

    No separate hold-out test set is used. Reported performance is based on
    repeated stratified CV. A final tree is trained on all run01 data using
    the best CV-selected parameters and saved, but this final model is not
    evaluated on the same data for reported performance.
    """

    if fc_types is None:
        fc_types = {
            "task_types": ["restAP", "restPA"],
            "band_types": ["slow3", "slow4", "slow5", "full"],
        }

    if param_grid is None:
        param_grid = {
            "max_depth": [1, 2],
            "min_samples_leaf": [2, 3],
            "ccp_alpha": [0.0, 0.01],
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

                model = FCDecisionTreeClassifier(
                    band_type=band_type,
                    task_type=task_type,
                    random_state=random_state,
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

        final_model = FCDecisionTreeClassifier(
            task_type=task_type,
            band_type=band_type,
            random_state=random_state,
            **best_params,
        )

        final_model.fit(X, y)
        final_model.save()

    cv_df = pd.DataFrame(cv_results)
    selected_df = pd.DataFrame(selected_results)

    report_dir = os.path.join(ARTIFACT_DIR, "reports", "DT")
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

class FCDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(
        self, 
        band_type: BandType, 
        task_type: TaskType, 
        max_depth: Optional[int] = None,
        min_samples_leaf: float = 1,
        ccp_alpha: float = 0.0,
        random_state=42,
        **kwargs
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
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

        depth = p["max_depth"] if p["max_depth"] is not None else "none"
        leaf = p["min_samples_leaf"] if p["min_samples_leaf"] is not None else 1

        alpha = f"{p['ccp_alpha']:.4f}".rstrip("0").rstrip(".")
        if alpha == "":
            alpha = "0"

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

        config_str = _make_config_string(config_info, param_keys)

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=lambda x: x.item())

    print(json.dumps(report["condition_summary"], indent=4))

    return summary_df

def _make_config_string(config_info, param_keys):
    config_parts = [
        str(config_info["task_type"]),
        str(config_info["band_type"]),
    ]

    name_map = {
        "n_estimators": "n",
        "max_depth": "depth",
        "min_samples_leaf": "min_smpl",
        "ccp_alpha": "ccp_a",
    }

    for key in param_keys:
        short_name = name_map.get(key, key)
        value = _format_json_value(config_info.get(key))
        config_parts.append(f"{short_name}={value}")

    return "[" + ", ".join(config_parts) + "]"

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