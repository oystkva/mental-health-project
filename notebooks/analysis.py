import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ( 
    TR,
    N_CPUs,
    #paths
    PROJECT_ROOT,
    MDD_LIST, 
    HC_LIST,
    DATA_DIR,
)
from src.data_loader import (
    load_subject_list,
    load_zFCs
) 
from src.functional_connectivity import (
    run_zFC_pipeline,
    run_bandpass_zFC_pipeline,
    fisher_z2r,
    calculate_subject_bold_zFC
)
from src.memd import run_memd_pipeline
from plots.plot_FC import (
    plot_FC_comparison,
    plot_FC
)


# def plot_all_channel_imfs_overlaid(file_name: str) -> None:
#     saved_data = np.load(os.path.join(processed_MEMD_dir, file_name))

#     # save plot of imfs with all channels overlaid
#     print(saved_data.shape)
#     num_imfs = saved_data.shape[0]
#     fig, axs = plt.subplots(num_imfs, 1, figsize=(10, 2*num_imfs))
#     for i in range(num_imfs):
#         axs[i].plot(saved_data[i].T)
#         axs[i].set_title(f'IMF {i+1}')
#     plt.tight_layout()
#     plt.savefig(os.path.join(processed_MEMD_dir, "plots", f"{file_name[:-4]}.svg", format="svg"))
#     plt.close()


#      restAP: 1   -   restPA: 2
which_tasks = 1
#       run01: 1   -   run02:  2   -   not specified: 0
which_runs = 0
# Schaefer400: 1   -  Yan2023: 2   -   not specified: 0
which_atlases = 2

run_HC = True
run_MDD = True

run_MEMD = False
run_zFC = True
run_bandpass_zFC = True

# tree params
seed = 42
max_depth = 8


if __name__ == "__main__":
    #region config globals
    if which_tasks == 1:
        task = 'restAP'
    elif which_tasks == 2:
        task = 'restPA'
    else:
        raise ValueError("Task case not valid. 1 (restAP) and 2 (restPA) are valid values.")
    if which_runs == 0:
        run = ''
    elif which_runs == 1:
        run = 'run01'
    elif which_runs == 2:
        run = 'run02'
    else:
        raise ValueError("Run case not valid. 0, 1, and 2 are valid values.")
    if which_runs == 0:
        run = ''
    elif which_atlases == 1:
        atlas = 'Schaefer400'
    elif which_atlases == 2:
        atlas = 'Yan2023'
    else:
        raise ValueError("Atlas case not valid. 0, 1 and 2 are valid values.")
    if run_HC and not run_MDD:
        group = 'HC'
    elif not run_HC and run_MDD:
        group = 'MDD'
    elif run_HC and run_MDD:
        group = ''
    else:
        raise ValueError("Invalid group case. run_HC and run_MDD cannot both be false")
    #endregion
    
    processed_MEMD_dir = os.path.join(DATA_DIR, "MEMD_processed")
    zFC_dir = os.path.join(DATA_DIR, "zFC_matrices")


    if run_MEMD:
        if run_HC:
            hc_subjects = load_subject_list(HC_LIST)
            if which_atlases in [0, 1]:
                run_memd_pipeline(
                    subject_list=hc_subjects, 
                    out_dir=processed_MEMD_dir,
                    group="HC", 
                    task_type=task,
                    cortical_atlas="Schaefer400",
                    n_parallels=N_CPUs,
                )
            if which_atlases in [0, 2]:
                run_memd_pipeline(
                    subject_list=hc_subjects, 
                    out_dir=processed_MEMD_dir,
                    group="HC", 
                    task_type=task,
                    cortical_atlas="Yan2023",
                    n_parallels=N_CPUs,
                )
        if run_MDD:
            mdd_subjects = load_subject_list(MDD_LIST)
            if which_atlases in [0, 1]:
                run_memd_pipeline(
                    subject_list=mdd_subjects, 
                    group="MDD", 
                    task_type=task,
                    cortical_atlas="Schaefer400",
                    out_dir=processed_MEMD_dir,
                    n_parallels=N_CPUs
                )   
            if which_atlases in [0, 2]:
                run_memd_pipeline(
                    subject_list=mdd_subjects, 
                    group="MDD", 
                    task_type=task,
                    cortical_atlas="Yan2023",
                    out_dir=processed_MEMD_dir,
                    n_parallels=N_CPUs
                )
    if run_zFC:
        if which_atlases in [0, 1]:
            run_zFC_pipeline(
                run_types=[[group, task, run, "Schaefer400"]], 
                task_type=task, 
                memd_dir=processed_MEMD_dir, 
                n_parallels=N_CPUs, 
                out_dir=zFC_dir, 
                TR=TR
            )
        if which_atlases in [0, 2]:
            run_zFC_pipeline(
                run_types=[[group, task, run, "Yan2023"]], 
                task_type=task, 
                memd_dir=processed_MEMD_dir, 
                n_parallels=N_CPUs, 
                out_dir=zFC_dir, 
                TR=TR
            )

    if run_bandpass_zFC:
        if which_atlases in [0, 1]:
            run_bandpass_zFC_pipeline(
                run_types=[[group, task, run, "Schaefer400"]], 
                task_type=task, 
                n_parallels=N_CPUs, 
                out_dir=zFC_dir, 
                TR=TR
            )
        if which_atlases in [0, 2]:
            run_bandpass_zFC_pipeline(
                run_types=[[group, task, run, "Yan2023"]], 
                task_type=task, 
                n_parallels=N_CPUs, 
                out_dir=zFC_dir, 
                TR=TR
            )
    
    # full = calculate_subject_bold_zFC("HC_NDAR_INVDK220VPQ_restAP_run01_Yan2023_BOLD_signals.h5", "/cluster/home/oystkva/project/code/data/zFC_matrices/Yan2023/restAP/")

    # full = np.load(os.path.join(DATA_DIR, "zFC_matrices/Yan2023/restAP/full_parcels/HC_NDAR_INVDK220VPQ_restAP_run01_Yan2023_zFC_full.npy"))
    # plot_FC(fisher_z2r(full), "INVBL733HBP_Full")
    # print(full)

    # slow3 = np.load(os.path.join(DATA_DIR, "zFC_matrices/Yan2023/restAP_bandpass/full_parcels/HC_NDAR_INVDK220VPQ_restAP_run01_Yan2023_zFC_bandpass_slow3.npy"))
    # plot_FC(fisher_z2r(slow3), "INVBL733HBP_Slow3")
    # slow4 = np.load(os.path.join(DATA_DIR, "zFC_matrices/Yan2023/restAP_bandpass/full_parcels/HC_NDAR_INVDK220VPQ_restAP_run01_Yan2023_zFC_bandpass_slow4.npy"))
    # plot_FC(fisher_z2r(slow4), "INVBL733HBP_Slow4")
    # slow5 = np.load(os.path.join(DATA_DIR, "zFC_matrices/Yan2023/restAP_bandpass/full_parcels/HC_NDAR_INVDK220VPQ_restAP_run01_Yan2023_zFC_bandpass_slow5.npy"))
    # plot_FC(fisher_z2r(slow5), "INVBL733HBP_Slow5")
    
    # slow3 = np.load(os.path.join(DATA_DIR, "zFC_matrices/Yan2023_a/restAP_bandpass/full_parcels/HC_NDAR_INVDK220VPQ_restAP_run01_Yan2023_zFC_bandpass_slow3.npy"))
    # plot_FC(fisher_z2r(slow3), "INVBL733HBP_a_Slow3")
    # slow4 = np.load(os.path.join(DATA_DIR, "zFC_matrices/Yan2023_a/restAP_bandpass/full_parcels/HC_NDAR_INVDK220VPQ_restAP_run01_Yan2023_zFC_bandpass_slow4.npy"))
    # plot_FC(fisher_z2r(slow4), "INVBL733HBP_a_Slow4")
    # slow5 = np.load(os.path.join(DATA_DIR, "zFC_matrices/Yan2023_a/restAP_bandpass/full_parcels/HC_NDAR_INVDK220VPQ_restAP_run01_Yan2023_zFC_bandpass_slow5.npy"))
    # plot_FC(fisher_z2r(slow5), "INVBL733HBP_a_Slow5")

    pass