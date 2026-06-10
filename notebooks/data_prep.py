import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    N_CPUs,
    FMRI_DIR,
    HC_LIST,
    MDD_LIST,
    DATA_DIR,
)
from src.parcel import (
    parcel_data, 
    make_gifs_schaefer_vs_bold,
    ATLAS_PATHS,
    ATLAS_PATHS_S400,
)

def compare_parcellations():
    """
    Compare two parcellations to see if they're identical. Load the data from the h5 files in /data/BOLD_signals_parcelled/Yan2023/ and /data/BOLD_signals_parcelled/Yan2023_a/
    and compare per subject per run sepcs. If the data is identical, print a message confirming this. If not, print a message indicating that the data differs and specify which subjects and runs differ.
    """
    import numpy as np
    from src.data_loader import load_h5_file
    parcellation_1_dir = os.path.join(DATA_DIR, "BOLD_signals_parcelled", "Yan2023", "restAP")
    parcellation_2_dir = os.path.join(DATA_DIR, "BOLD_signals_parcelled", "Schaefer400", "restAP")

    subjects = set()
    for filename in os.listdir(parcellation_1_dir):
        if filename.endswith(".h5"):
            subject_key = filename.split("_")[0]+"_"+filename.split("_")[1]+"_"+filename.split("_")[2]
            subjects.add(subject_key)

    differences = []
    for subject in subjects:
        for run in ["restAP_run01", "restAP_run02"]:
            file_1 = os.path.join(parcellation_1_dir, f"{subject}_{run}_Yan2023_BOLD_signals.h5")
            file_2 = os.path.join(parcellation_2_dir, f"{subject}_{run}_Schaefer400_BOLD_signals.h5")
            if os.path.exists(file_1) and os.path.exists(file_2):
                dict_1 = load_h5_file(file_1)
                dict_2 = load_h5_file(file_2)
                key = next(iter(dict_1.keys()))
                data_1, data_2 = dict_1[key], dict_2[key]
                if not np.array_equal(data_1, data_2):
                    differences.append((subject, run))
                else:
                    print(f"{subject} {run} data is identical between the two parcellations.")
            elif not os.path.exists(file_1) and os.path.exists(file_2):
                print(f"{file_1} missing")
            elif os.path.exists(file_1) and not os.path.exists(file_2):
                print(f"{file_2} missing")
            else:
                print("both mssing")
                print(file_1)
                print(file_2)
                print("_----")

    if not differences:
        print("The parcellated data from the two parcellations is identical for all subjects and runs.")
    else:
        print(f"The parcellated data differs between the two parcellations for the following {len(differences)} subjects and runs:")
        for subject, run in differences:
            print(f"Subject: {subject}, Run: {run}")

run_HC = not True
run_MDD = not True

run_restAP = not True
run_restPA = not True

atlas_paths = ATLAS_PATHS # ATLAS_PATHS = Yan2023 config, ATLAS_PATHS_S400 = Schaefer400 config

if __name__ == "__main__":

    # make_gifs_schaefer_vs_bold(
    #     bold_path=os.path.join(FMRI_DIR,  "sub-NDARINVWU297KRB", "func", "sub-NDARINVWU297KRB_task-restAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"),
    # )

    compare_parcellations()

    save_dir = os.path.join(DATA_DIR, "BOLD_signals_parcelled")

    if run_restAP:
        if run_HC:
            parcel_data(
                data_dir=FMRI_DIR,
                out_dir=save_dir,
                subject_list=HC_LIST,
                subject_group="HC",
                n_parallels=N_CPUs,
                atlas_paths=atlas_paths,
                fmri_run_types=[
                    ["task-restAP", "run-02", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
                ],
            )
        if run_MDD:
            parcel_data(
                data_dir=FMRI_DIR,
                out_dir=save_dir,
                subject_list=MDD_LIST,
                subject_group="MDD",
                n_parallels=N_CPUs,
                atlas_paths=atlas_paths,
                fmri_run_types=[
                    ["task-restAP", "run-02", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
                ]
            )
    if run_restPA:
        if run_HC:
            parcel_data(
                data_dir=FMRI_DIR,
                out_dir=save_dir,
                subject_list=HC_LIST,
                subject_group="HC",
                n_parallels=N_CPUs,
                atlas_paths=atlas_paths,
                fmri_run_types=[
                    ["task-restPA", "run-02", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
                ],
            )
        if run_MDD:
            parcel_data(
                data_dir=FMRI_DIR,
                out_dir=save_dir,
                subject_list=MDD_LIST,
                subject_group="MDD",
                n_parallels=N_CPUs,
                atlas_paths=atlas_paths,
                fmri_run_types=[
                    ["task-restPA", "run-02", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
                ]
            )