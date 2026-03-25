import sys, os
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = "/cluster/home/oystkva/project/code"

from models.decision_tree import train_tree

from src.data_loader import load_zFC_df



if __name__ == "__main__":

    train_tree()