import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

PROJECT_ROOT = "/cluster/home/oystkva/project/code"
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "plots"))

from config import ( 
    TR,
    N_CPUs,
    #paths
    MDD_LIST, 
    HC_LIST,
    DATA_DIR,
    #endregion
)
from data_loader import load_subject_list 
from functional_connectivity import run_zFC_pipeline, run_bandpass_zFC_pipeline
from memd import run_memd_pipeline



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




if __name__ == "__main__":
    
    processed_MEMD_dir = os.path.join(DATA_DIR, "MEMD_processed")
    zFC_dir = os.path.join(DATA_DIR, "zFC_matrices")

    # hc_subjects = load_subject_list(HC_LIST)
    # run_memd_pipeline(
    #     subject_list=hc_subjects, 
    #     group="HC", 
    #     task_type="restAP",
    #     cortical_atlas="Yan2023",
    #     out_dir=processed_MEMD_dir,
    #     n_parallels=N_CPUs,
    # )

    # mdd_subjects = load_subject_list(MDD_LIST)
    # run_memd_pipeline(
    #     subject_list=mdd_subjects, 
    #     group="MDD", 
    #     task_type="restAP",
    #     cortical_atlas="Yan2023",
    #     out_dir=processed_MEMD_dir,
    #     n_parallels=N_CPUs
    # )

    #region plotting (TODO: remove (?))
    # # Parallel(n_jobs=N_CPUs)(
    # #     delayed(plot_all_channel_imfs_overlaid)(file_name) for file_name in tqdm(imf_paths))
    
    # # # candidate_nr = 1
    # # # slice_start = 300
    # # # slice_stop = slice_start + 10

    # # # fig, axs = plt.subplots(len(band_signals)+1, 1, figsize=(10, 4+4*len(band_signals)))
    # # # for band_name, signal in band_signals.items():
    # # #     if signal is not None:
    # # #         plt.figure(figsize=(10, 4))
    # # #         time = TR * np.arange(signal.shape[1])
    # # #         plt.plot(time[:50], signal[candidate_nr][:50].T)  # plot first 200 timepoints
    # # #         plt.title(f"{band_name} signal - Channel 1")
    # # #         plt.xlabel("Time [s]")
    # # #         plt.ylabel("Signal Amplitude")
    # # #         plt.grid(True)
    # # #         plt.savefig(os.path.join(PROJECT_ROOT, f"{imf_paths[candidate_nr][:-4]}_{band_name}_signal.svg"), format="svg")
    # # #         plt.close()

    # # #         time = TR * np.arange(signal.shape[1])
    # # #         axs[list(band_signals.keys()).index(band_name)].plot(time[slice_start:slice_stop], signal[candidate_nr][slice_start:slice_stop].T)
    # # #         axs[list(band_signals.keys()).index(band_name)].set_title(f"{band_name} signal - Channel 1")
    # # #         axs[list(band_signals.keys()).index(band_name)].set_xlabel("Time [s]")
    # # #         axs[list(band_signals.keys()).index(band_name)].set_ylabel("Signal Amplitude")
    # # #         axs[list(band_signals.keys()).index(band_name)].grid(True)
    # # # temp = bold_signals[list(bold_signals.keys())[0]]
    # # # time = TR * np.arange(temp.shape[1])
    # # # axs[-1].plot(time[slice_start:slice_stop], temp[candidate_nr][slice_start:slice_stop].T)
    # # # axs[-1].set_title("Original BOLD signal - Channel 1")
    # # # axs[-1].set_xlabel("Time [s]")
    # # # axs[-1].set_ylabel("Signal Amplitude")
    # # # axs[-1].grid(True)
    # # # plt.tight_layout()
    # # # print(imf_paths[candidate_nr][:-9])
    # # # plt.savefig(os.path.join(PROJECT_ROOT, f"{imf_paths[candidate_nr][:-9]}_slow_band_signals.svg"), format="svg")
    # # # plt.close()
    # endregion

    #region plotting FC comparisons
    # # plot_FC_comparison(
    # #     [FC, FC_slow5, FC_slow4, FC_slow3], 
    # #     ["Original BOLD FC", "Slow-5 FC", "Slow-4 FC", "Slow-3 FC"],
    # #     filename = f"{imf_paths[candidate_nr][:-9]}_parcel_band_comparison",
    # # )


    # # plot_FC_comparison(
    # #     [FC_mean, FC_slow5_mean, FC_slow4_mean, FC_slow3_mean], 
    # #     ["Original BOLD Mean Network FC", "Slow-5 Mean Network FC", "Slow-4 Mean Network FC", "Slow-3 Mean Network FC"],
    # #     filename = f"{imf_paths[candidate_nr][:-9]}_network_band_comparison",
    # #     networks=True,
    # # )
    #endregion

    # run_zFC_pipeline(
    #     [["", "restAP", "", "Schaefer400"]], 
    #     task_type="restAP", 
    #     memd_dir=processed_MEMD_dir, 
    #     n_parallels=N_CPUs, 
    #     out_dir=zFC_dir, 
    #     TR=TR
    # )
    
    pass