import sys, os
import json
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = "/cluster/home/oystkva/project/code"

from models.random_forest import FCRandomForestClassifier, crossval_fc_forest, plot_hyp_param_res_comparison

from models.graph_neural_net import FCGCN, load_fc_graph_dataset

from models.mlp import FCMLP, make_zFC_loader

from src.data_loader import load_zFC_df
from src.utils import set_seed


# restAP: 1 - restPA: 2 - all: 3
task_case = 1
# full: 1 - all: 2 - slow3: 3 - slow4: 4 - slow5 - 5
slow_case = 5
both_runs = True
n_trees = 100

# nn params
n_epochs = 40

# tree params
split_seed = 42
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
#endregion

def test_random_forest(b, t):
    RFclf = FCRandomForestClassifier(
        band_type=b, 
        task_type=t, 
        n_estimators=n_trees,
        max_depth=max_depth, 
        random_state=split_seed, 
        ccp_alpha=ccp_alpha,
        min_samples_split=5,
    )

    data = load_zFC_df(band_type=RFclf.band_type, task_type=RFclf.task_type, include_all_runs=both_runs)
    X = data.iloc[:, 0:-1]
    y = data.MDD

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=split_seed)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    RFclf.fit(X_train, y_train)
    RFclf.evaluate(X_test, y_test)
    RFclf.save()

def run_train(t, b):
    dataset = load_fc_graph_dataset(
            task_type=t, 
            band_type=b, 
            include_all_runs=both_runs,
            atlas_type="Schaefer400",
            network_means=False,
            decomp_method="memd",
            threshold=0.5,
        )

    seed = 42

    labels = [graph.y.item() for graph in dataset]

    set_seed(seed)

    train_val_graphs, test_graphs = train_test_split(
        dataset,
        test_size=0.15,
        random_state=42,
        stratify=labels,
    )

    train_graphs, val_graphs = train_test_split(
        train_val_graphs,
        test_size=(0.15/0.85),  # 0.25 x 0.8 = 0.2
        random_state=42,
        stratify=[int(graph.y.item()) for graph in train_val_graphs],
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_graphs,
        batch_size=7,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_graphs,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_graphs,
        shuffle=False,
    )

    Parallel(n_jobs=6)(
        delayed(run_train_train_loop)(
            t,
            b,
            train_loader,
            val_loader,
            c,
        ) for c in ['CERP']
    )

    
def run_train_train_loop(
        t,
        b,
        train_loader,
        val_loader,
        c,
):
    set_seed(seed)

    model = FCGCN(
        task=t,
        band=b,
        atlas="Schaefer400",
        num_node_features=434,
    )

    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=n_epochs,
        # lr=0.0001,
        # label_smoothing=0.05,
        log_every=5,
        crit=c
    )

    with open(os.path.join(str(model.model_dir), "_history.json"), 'w') as f:
        json.dump(
            obj=history,
            fp=f,
            indent=4,
        )
    
    model.save()


if __name__ == "__main__":

    # crossval_fc_forest(
    #     # fc_types={
    #     #     'task_types': ['restAP'],
    #     #     'band_types': ['slow5']
    #     # },
    #     # param_grid={
    #     #     'n_estimators':     [10],
    #     #     'max_depth':        [5],
    #     #     'min_samples_leaf': [3],
    #     #     'ccp_alpha':        [0.0],
    #     # },
    #     param_grid={
    #         'n_estimators':     [100, 200, 300],
    #         'max_depth':        [1, 2, 3, 4],
    #         'min_samples_leaf': [1, 2, 3, 5],
    #     },
    # )
    from itertools import product
    t_list = ['restPA']
    b_list = ['slow3']

    run_train('restPA', 'slow3')

    # for t in ['restAP', 'restPA']:
    #     for b in ['full', 'slow3', 'slow4', 'slow5']:
        # # for b in ['slow5']:
        #     dataset = load_fc_graph_dataset(
        #         task_type=t, 
        #         band_type=b, 
        #         include_all_runs=both_runs,
        #         atlas_type="Schaefer400",
        #         network_means=False,
        #         decomp_method="memd",
        #         threshold=0.2,
        #     )

        #     labels = [graph.y.item() for graph in dataset]

        #     train_val_graphs, test_graphs = train_test_split(
        #         dataset,
        #         test_size=0.15,
        #         random_state=42,
        #         stratify=labels,
        #     )

        #     train_graphs, val_graphs = train_test_split(
        #         train_val_graphs,
        #         test_size=(0.15/0.85),  # 0.25 x 0.8 = 0.2
        #         random_state=42,
        #         stratify=[int(graph.y.item()) for graph in train_val_graphs],
        #     )

        #     train_loader = DataLoader(
        #         train_graphs,
        #         batch_size=7,
        #         shuffle=True,
        #     )

        #     val_loader = DataLoader(
        #         val_graphs,
        #         shuffle=False,
        #     )

        #     test_loader = DataLoader(
        #         test_graphs,
        #         shuffle=False,
        #     )

        #     model = FCGCN(
        #         task=t,
        #         band=b,
        #         atlas="Schaefer400",
        #         num_node_features=434,
        #     )

        #     history = model.fit(
        #         train_loader=train_graphs,
        #         val_loader=val_graphs,
        #         epochs=n_epochs,
        #         label_smoothing=0.05,
        #         log_every=10,
        #     )

        #     with open(os.path.join(model.model_dir, "history.json"), 'w') as f:
        #         json.dump(
        #             obj=history,
        #             fp=f,
        #             indent=4,
        #         )
            
        #     model.save()

    # data = load_zFC_df(band_type='slow5', task_type='restAP', include_all_runs=both_runs, network_means=False)
    # X = data.iloc[:, 0:-1]
    # y = data.MDD

    # train_df, temp_df = train_test_split(
    #     data,
    #     test_size=0.30,
    #     random_state=seed,
    #     stratify=data["MDD"],
    # )

    # val_df, test_df = train_test_split(
    #     temp_df,
    #     test_size=0.50,
    #     random_state=seed,
    #     stratify=temp_df["MDD"],
    # )

    # scaler = StandardScaler()

    # train_df = train_df.copy()
    # val_df = val_df.copy()
    # test_df = test_df.copy()

    # feature_cols = train_df.columns[:-1]
    # train_df.loc[:, feature_cols] = scaler.fit_transform(train_df.loc[:, feature_cols])
    # val_df.loc[:, feature_cols] = scaler.transform(val_df.loc[:, feature_cols])
    # test_df.loc[:, feature_cols] = scaler.transform(test_df.loc[:, feature_cols])

    # train_loader = make_zFC_loader(train_df)
    # val_loader = make_zFC_loader(val_df)
    # test_loader = make_zFC_loader(test_df)

    # model = FCMLP(
    #     task='restPA',
    #     band='slow4',
    #     input_dim=int(435*434/2),
    #     hidden_dim=16,
    # )    

    # history = model.fit(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=70,
    #     lr=0.0001
    # )

    # with open(os.path.join(str(model.model_dir), "_history.json"), 'w') as f:
    #     json.dump(
    #         obj=history,
    #         fp=f,
    #         indent=4,
    #     )
# plot_hyp_param_res_comparison()