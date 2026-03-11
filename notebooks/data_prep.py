import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
print(f"Project root set to: {PROJECT_ROOT}")

from config import (
    N_CPUs,
    FMRI_DIR,
    HC_LIST,
    MDD_LIST,
    DATA_DIR,
)
from parcel import parcel_data

if __name__ == "__main__":

    save_parcel_data = os.path.join(DATA_DIR, "BOLD_signals_parcelled")

    parcel_data(
        data_dir=FMRI_DIR,
        out_dir=save_parcel_data,
        subject_list=HC_LIST,
        subject_group="HC",
        n_parallels=N_CPUs,
        fmri_run_types=[
                    ["task-restAP", "run-01", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
                ]
    )

    parcel_data(
        data_dir=FMRI_DIR,
        out_dir=save_parcel_data,
        subject_list=MDD_LIST,
        subject_group="MDD",
        n_parallels=N_CPUs,
        fmri_run_types=[
                    ["task-restAP", "run-01", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
                ]
    )
