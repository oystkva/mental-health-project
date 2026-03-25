import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

from src.config import (
    N_CPUs,
    FMRI_DIR,
    HC_LIST,
    MDD_LIST,
    DATA_DIR,
)
from src.parcel import parcel_data

if __name__ == "__main__":

    save_parcel_data = os.path.join(DATA_DIR, "BOLD_signals_parcelled")

    parcel_data(
        data_dir=FMRI_DIR,
        out_dir=save_parcel_data,
        subject_list=HC_LIST,
        subject_group="HC",
        n_parallels=N_CPUs,
        fmri_run_types=[
                    ["task-restAP", "run-02", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
                ]
    )

    parcel_data(
        data_dir=FMRI_DIR,
        out_dir=save_parcel_data,
        subject_list=MDD_LIST,
        subject_group="MDD",
        n_parallels=N_CPUs,
        fmri_run_types=[
                    ["task-restAP", "run-02", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
                ]
    )
