import os, sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pydotplus

sys.path.append(str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = "/cluster/home/oystkva/project/code"

from src.data_loader import load_zFC_df

def train_tree():
    data = load_zFC_df('slow4', 'restPA')
    X = data.iloc[:, 0:-1]
    y = data.MDD

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X.columns)
    print(X_train)
    tree = DecisionTreeClassifier(random_state=1)

    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score = {f1}')


    # Export tree to graphviz format
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=X.columns,
        class_names=["HC", "MDD"],   # 0 = HC, 1 = MDD
        filled=True,
        rounded=True,
        special_characters=True
    )


    # Highlight decision path for one test sample
    sample_idx = np.random.randint(0, len(X_test), 3)
    print(sample_idx)
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
        sample = X_test.iloc[[idx]]   # keep as DataFrame
        decision_path = tree.decision_path(sample)

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

        filename = "tree"+str(k)+".png"
        graph.write_png(filename)

        print(f"Tree visualization saved to {filename}")