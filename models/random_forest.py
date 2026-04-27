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
        'n_estimators':     [10, 25, 50, 100],
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

                model = FCRandomForestClassifier(
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

        model = FCRandomForestClassifier(
            task_type=task_type,
            band_type=band_type,
            **best_params,
        )
        model.fit(X_t, y_t)
        model.evaluate(X_test, y_test)
        model.save()

    df = pd.DataFrame(results)
    return df.sort_values('mean_f1', ascending=False).reset_index(drop=True)


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
        return f"tree_{self.task_type}_{self.band_type}_n{n_estimators}_f{max_features}_d{depth}_l{leaf}"

    def _build_info(self) -> dict:
        has_scores = hasattr(self, 'eval_scores_')
        return {
            "identity": {
                "task_type": str(self.task_type),
                "band_type": str(self.band_type),
            },
            "hyperparameters": self.get_params(),
            "eval_scores": (
                {k: round(v, 4) for k, v in self.eval_scores_.items()}
                if has_scores else "not evaluated — call evaluate() first"
            ),
            "saved_at": datetime.now().isoformat(timespec='seconds') + "Z",
        }