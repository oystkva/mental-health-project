import sys, os
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = "/cluster/home/oystkva/project/code"

from models.random_forest import FCRandomForestClassifier, crossval_fc_forest, plot_hyp_param_res_comparison

from src.data_loader import load_zFC_df

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# restAP: 1 - restPA: 2 - all: 3
task_case = 2
# full: 1 - all: 2 - slow3: 3 - slow4: 4 - slow5 - 5
slow_case = 5
both_runs = True
n_trees = 100

# tree params
seed = 42
max_depth = None
ccp_alpha = 0.0

#region config globals
if task_case == 1:
    task_type = 'restAP'
elif task_case == 2:
    task_type = 'restPA'
elif task_case == 3:
    task_type = 'all'
else:
    raise ValueError("Task case not valid. 1, 2 and 3 are valid values.")
if slow_case == 1:
    band = 'full'
elif slow_case == 2:
    band = 'all'
elif slow_case in [3, 4, 5]:
    band = 'slow' + str(slow_case)
else:
    raise ValueError("Band case not valid. 1, 2, 3, 4 and 5 are valid values.")
runs = 2 if both_runs else 1
#endregion

def test_random_forest(b, t):
    RFclf = FCRandomForestClassifier(
        band_type=b, 
        task_type=t, 
        n_estimators=n_trees,
        max_depth=max_depth, 
        random_state=seed, 
        ccp_alpha=ccp_alpha,
        min_samples_split=5,
    )

    data = load_zFC_df(band_type=RFclf.band_type, task_type=RFclf.task_type, runs=1)
    X = data.iloc[:, 0:-1]
    y = data.MDD

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X.columns)

    RFclf.fit(X_train, y_train)
    RFclf.evaluate(X_test, y_test)
    RFclf.save()


if __name__ == "__main__":

    crossval_fc_forest(
        # fc_types={
        #     'task_types': ['restAP'],
        #     'band_types': ['slow5']
        # },
        # param_grid={
        #     'n_estimators':     [10],
        #     'max_depth':        [5],
        #     'min_samples_leaf': [3],
        #     'ccp_alpha':        [0.0],
        # },
        param_grid={
            'n_estimators':     [100, 200, 300],
            'max_depth':        [1, 2, 3, 4],
            'min_samples_leaf': [1, 2, 3, 5],
        },
    )

    # plot_hyp_param_res_comparison()