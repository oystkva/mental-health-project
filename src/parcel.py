import os, sys
import numpy as np
from io import BytesIO
import nibabel as nib
from nibabel.processing import resample_from_to
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import resample_to_img, new_img_like
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image

from src.config import PROJECT_ROOT, FMRI_DIR
from src.data_loader import (
    load_subject_list, 
    list_fmri_nii_file_paths, 
    save_BOLD_signals_h5,
)

Buckner7_PATH = os.path.join(PROJECT_ROOT, "src/brain_atlases/atl-Buckner7_space-MNI_dseg.nii")

ATLAS_PATHS = {
    "Schaefer400": os.path.join(PROJECT_ROOT, "src/brain_atlases/Schaefer2018_400Parcels_Kong2022_17Networks_order_FSLMNI152_2mm.nii.gz"),
    "Tian_Subcortex": os.path.join(PROJECT_ROOT, "src/brain_atlases/Tian_Subcortex_S2_3T.nii.gz"),
    "BucknerLR": os.path.join(PROJECT_ROOT, "src/brain_atlases/atl-BucknerLR_space-MNI_dseg.nii"),
}

ATLAS_PATHS_YAN = {
    "Yan2023": os.path.join(PROJECT_ROOT, "src/brain_atlases/Yan2023_homotopic_400Parcels_Kong2022_17Networks.dlabel.nii"),
    "Tian_Subcortex": os.path.join(PROJECT_ROOT, "src/brain_atlases/Tian_Subcortex_S2_3T.nii.gz"),
    "BucknerLR": os.path.join(PROJECT_ROOT, "src/brain_atlases/atl-BucknerLR_space-MNI_dseg.nii"),
}

def make_BucknerLR(
        buckner_path: str = Buckner7_PATH,
        out_path: str = ATLAS_PATHS["BucknerLR"],
):
    """
    Convert the Buckner7 atlas with 7 regions into a left-right hemisphere atlas with 2 regions.

    Args:
        buckner_path (str): Path to the input Buckner7 atlas NIfTI file.
        out_path (str): Path to save the output BucknerLR atlas NIfTI file. 
    """
    buckner7_img = nib.load(buckner_path)
    buckner7_data = buckner7_img.get_fdata()
    buckner7_labels = buckner7_data.astype(int)
    affine = buckner7_img.affine

    X, Y, Z = buckner7_labels.shape

    print(X, Y, Z)
    cerebellar_mask = buckner7_labels > 0

    i, j, k = np.meshgrid(
        np.arange(X), np.arange(Y), np.arange(Z), 
        indexing="ij"
    )

    ijk = np.vstack([
        i.ravel(),
        j.ravel(),
        k.ravel(),
        np.ones(i.size)
    ])

    xyz = affine @ ijk
    x_coords = xyz[0, :].reshape(X, Y, Z)

    left_mask = (x_coords <= 0) & cerebellar_mask
    right_mask = (x_coords > 0) & cerebellar_mask

    lr_mask = np.zeros_like(buckner7_labels, dtype=np.int16)
    lr_mask[left_mask] = 1
    lr_mask[right_mask] = 2

    lr_img = nib.Nifti1Image(lr_mask, affine, header=buckner7_img.header)
    nib.save(lr_img, out_path)

    unique_labels = np.unique(lr_mask)
    print(f"Saved BucknerLR atlas to {out_path} with labels: {unique_labels}")

def make_gifs_of_atlas_slices():
    """
    Help function to create gifs of axial, coronal, and sagittal slices of the BucknerLR atlas to visualize the regions.
    """
    # Create axial slice gifs for BucknerLR atlas in x, y, z directions adn compare with Buckner7
    atlas_img = nib.load(ATLAS_PATHS["BucknerLR"])
    atlas_data = atlas_img.get_fdata()
    X, Y, Z = atlas_data.shape
    slices = {
        "axial": [atlas_data[:, :, z] for z in range(0, Z, 5)],
        "coronal": [atlas_data[:, y, :] for y in range(0, Y, 5)],
        "sagittal": [atlas_data[x, :, :] for x in range(0, X, 5)],
    }

    atlas7_img = nib.load(Buckner7_PATH)
    atlas7_data = atlas7_img.get_fdata()
    slices7 = {
        "axial": [atlas7_data[:, :, z] for z in range(0, Z, 5)],
        "coronal": [atlas7_data[:, y, :] for y in range(0, Y, 5)],
        "sagittal": [atlas7_data[x, :, :] for x in range(0, X, 5)],
    }


    # Create images for each atlas and vstack them
    for direction, slice_list in slices.items():
        images = []
        for s in slice_list:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(np.rot90(s), cmap="plasma", vmin=0, vmax=7)
            ax[0].set_title("BucknerLR")
            ax[0].axis("off")
            
            # Get corresponding slice from Buckner7
            s7 = slices7[direction].pop(0)
            ax[1].imshow(np.rot90(s7), cmap="plasma", vmin=0, vmax=7)
            ax[1].set_title("Buckner7")
            ax[1].axis("off")

            plt.suptitle(f"{direction.capitalize()} Slice", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)

            # Save to a temporary buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            images.append(Image.open(buf))
            plt.close()

        # Save as GIF
        gif_path = f"bucknerLR_{direction}_slices.gif"
        images[0].save(
            fp=gif_path,
            format='GIF',
            append_images=images[1:],
            save_all=True,
            duration=500,
            loop=0
        )
        print(f"Saved GIF: {gif_path}")

def create_mask_image(atlas_path: str, bold_path: str) -> nib.Nifti1Image:
    atlas_img = nib.load(atlas_path)
    bold_img = nib.load(bold_path)

    resampled_atlas = resample_to_img(atlas_img, bold_img, interpolation="nearest")
    resampled_data = resampled_atlas.get_fdata()

    mask_data = (resampled_data > 0).astype(np.int16)
    mask_img = new_img_like(bold_img, mask_data, copy_header=True)
    return mask_img


def parcellate_to_BOLD(
    bold_path: str,
    atlas_paths: dict =ATLAS_PATHS,
) -> np.ndarray:
    """
    Parcellate one fMRIPrep run into a BOLD time series for each ROI.
    Args:
        bold_path (str): Path to the fMRIPrep preprocessed BOLD NIfTI file.
        atlas_paths (dict): Dictionary of atlas names and their NIfTI file paths.
    Returns:
        np.ndarray: 2D array of shape (n_timepoints, n_rois) with the parcellated BOLD time series.
    """
    ROIs = []
    for atlas in atlas_paths:
        mask = NiftiLabelsMasker(
            labels_img=atlas_paths[atlas],
            mask_img=create_mask_image(atlas_paths[atlas], bold_path),
        )
        time_series = mask.fit_transform(bold_path).T
        ROIs.append(time_series)

    return np.vstack(ROIs)

def load_subject_BOLD_signals(
    runpaths: list[str],
) -> dict:
    """
    Load and parcellate BOLD signals for a list of fMRIPrep run file paths.
    Args:
        runpaths (list[str]): List of fMRIPrep preprocessed BOLD NIfTI file paths.
        atlas_paths (dict): Dictionary of atlas names and their NIfTI file paths.
    Returns:
        dict: Dictionary with run file paths as keys and parcellated BOLD time series as values.
    """
    bold_signals = {}
    for runpath in runpaths:
        bold_signals[runpath] = parcellate_to_BOLD(bold_path=runpath)
    
    print(f"Parcellated {len(runpaths)} runs.")
    return bold_signals

def parcellate_subject(
    subjectkey: str,
    runs: list[str],
    grouping: str,
    out_dir: str,
    TR: float = 0.8,
    atlas_paths: dict = ATLAS_PATHS,
):
    """
    Parcellate all fMRIPrep runs for one subject and save to HDF5.
    Args:
        subjectkey (str): Subject key.
        runs (list[str]): List of fMRIPrep preprocessed BOLD NIfTI file paths for the subject.
        grouping (str): Subject grouping (e.g., "HC" or "MDD").
        out_dir (str): Directory to save the .h5 file.
        TR (float): Repetition time in seconds.
        atlas_paths (dict): Dictionary of atlas names and their NIfTI file paths.
    """
    atlas_type = "Yan2023" if "Yan2023" in atlas_paths.keys() else "Schaefer400"

    run_num = "run01" if "run-01" in runs[0] else "run02"
    task = "restPA" if "restPA" in runs[0] else "restAP"

    out_path = os.path.join(out_dir, task, f"{grouping}_{subjectkey}_{task}_{run_num}_{atlas_type}_BOLD_signals.h5")

    if os.path.exists(out_path):
        print(f"[SKIP]: [{grouping}] {subjectkey} already exists.")
        return
    
    try:
        bold_signals = load_subject_BOLD_signals(runpaths=runs)
        save_BOLD_signals_h5(
            bold_signals=bold_signals,
            out_dir=out_path,
            subjectkey=subjectkey,
            atlas_paths=atlas_paths,
            TR=TR,
            n_runs=len(runs),
        )
        print(f"[DONE]: [{grouping}] {subjectkey} {task}")

    except Exception as e:
        print(f"[ERROR]: [{grouping}] {subjectkey} {task} failed with error: {e}")


def parcel_data(
    data_dir: str,
    out_dir: str,
    subject_list: str,
    subject_group: str,
    TR: float = 0.8,
    n_parallels: int = 1,
    atlas_paths: dict = ATLAS_PATHS,
    fmri_run_types=[
                ["task-restPA", "run-01", "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"],
            ]
):
    """
    Parcellate fMRI data for a list of subjects and save the BOLD time series to HDF5 files.
    Args:
        data_dir (str): Directory containing fMRIPrep preprocessed data.
        out_dir (str): Directory to save the .h5 files.
        subject_list (str): Path to the text file with the list of subject keys.
        subject_group (str): Subject grouping (e.g., "HC" or "MDD").
        TR (float): Repetition time in seconds.
        n_parallels (int): Number of parallel jobs to run.
        atlas_paths (dict): Dictionary of atlas names and their NIfTI file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    subjectkeys = load_subject_list(subject_list)
    print("Loaded subject list with", len(subjectkeys), "subjects from the file:", subject_list)

    run_dict = {}
    for subjectkey in tqdm(subjectkeys):
        run_dict[subjectkey] = list_fmri_nii_file_paths(
            data_dir=data_dir,
            subjectkey=subjectkey,
            fmri_run_types=fmri_run_types
        )
    print("Collected fMRI run paths for all subjects.")

    Parallel(n_jobs=n_parallels)(
        delayed(parcellate_subject)(
            subjectkey,
            runs,
            subject_group,
            out_dir,
            TR,
            atlas_paths,
        )
        for subjectkey, runs in tqdm(run_dict.items(), desc=f"Parcellating subject {subject_group}")
    )

    return


# # for key in ATLAS_PATHS:
# #     atlas = nib.load(ATLAS_PATHS[key])
# #     labels = np.unique(atlas.get_fdata()).astype(int)
# #     print(f"Atlas: {key}")
# #     print("labels:", labels)
# #     print("Number of non-zero labels in the atlas:", len(labels[labels > 0]))
# #     print("-----")

# ## Create png of Buckner atlas slices for visualization with a subplot for each label


# fmri_df = nib.load(os.path.join(FMRI_DIR, "sub-NDARINVWU297KRB", "func", "sub-NDARINVWU297KRB_task-restAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"))

# atlas = nib.load(ATLAS_PATHS["Buckner7"])

# atlas_resampled = resample_from_to(
#         atlas,
#         (fmri_df.shape[:3], fmri_df.affine),
#         order=0,  # nearest neighbour so labels stay integer
#     )

# data = fmri_df.dataobj
# atlas_data = atlas_resampled.get_fdata()
# slices = [5, 10, 15, 20, 25]

# X, Y, Z, T = data.shape

# # Compute world (MNI) x-coordinates for each voxel using the atlas affine

# affine = fmri_df.affine  # (4, 4)
# i, j, k = np.meshgrid(
#     np.arange(X), np.arange(Y), np.arange(Z), indexing="ij"
# )
# ijk = np.vstack([
#     i.ravel(),
#     j.ravel(),
#     k.ravel(),
#     np.ones(i.size)
# ])  # (4, N_voxels)

# xyz = affine @ ijk  # (4, N_voxels)
# x_coords = xyz[0, :].reshape(X, Y, Z)  # (X, Y, Z)

# cereb_mask = atlas_data > 0
# left_hemi_mask = (x_coords < 0) & cereb_mask
# right_hemi_mask = (x_coords > 0) & cereb_mask


# for s in tqdm(slices):
#     left_data = data[:, :, s, 200]
#     right_data = data[:, :, s, 200]
#     cereb_masked_data = data[:, :, s, 200]
#     left_data = np.where(left_hemi_mask[:, :, s]*data[:, :, s, 200], 1, 0)
#     right_data = np.where(right_hemi_mask[:, :, s]*data[:, :, s, 200], 2, 0)
#     cereb_masked_data = left_data + right_data
#     print(left_data.shape, right_data.shape, cereb_masked_data.shape)

#     plt.figure(figsize=(10, 5))
#     plt.suptitle(f"Buckner7 Atlas - Axial Slice z={s}", fontsize=16)

#     plt.subplot(1, 3, 1)
#     plt.title("Left Hemisphere")
#     plt.imshow(left_data, cmap="tab20")
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.title("Right Hemisphere")
#     plt.imshow(right_data, cmap="tab20")
#     plt.axis("off")

#     plt.subplot(1, 3, 3)
#     plt.title("Both Hemispheres")
#     plt.imshow(cereb_masked_data, cmap="tab20")
#     plt.axis("off")

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(f"buckner7_atlas_slice_z{s}.png", dpi=150)

def make_gifs_schaefer_vs_bold(bold_path, atlas_path=None, out_prefix="schaefer_vs_bold"):
    """
    Create GIFs of axial, coronal, and sagittal slices comparing the Schaefer atlas
    and a single fMRI image (mean over time).

    Parameters
    ----------
    bold_path : str
        Path to a 4D (or 3D) fMRI NIfTI image.
    atlas_path : str or None
        Path to the Schaefer atlas NIfTI. If None, uses ATLAS_PATHS["Schaefer400"].
    out_prefix : str
        Prefix for the output GIF filenames.
    """
    if atlas_path is None:
        atlas_path = ATLAS_PATHS["Schaefer400"]

    # --- Load atlas ---
    atlas_img = nib.load(atlas_path)

    # --- Load bold and make a single 3D image (mean over time) ---
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()

    if bold_data.ndim == 4:
        bold_mean = bold_data.mean(axis=3)  # (X, Y, Z)
    else:
        bold_mean = bold_data  # already 3D

    # --- Resample atlas to BOLD space (so shapes match for slicing) ---
    if resample_to_img is not None:
        atlas_res = resample_to_img(atlas_img, bold_img, interpolation="nearest")
    else:
        # Fallback: assume same space/shape
        atlas_res = atlas_img

    atlas_data = atlas_res.get_fdata()
    X, Y, Z = atlas_data.shape

    # --- Build slice lists for each orientation ---
    slices_atlas = {
        "axial":    [atlas_data[:, :, z] for z in range(0, Z, 5)],
        "coronal":  [atlas_data[:, y, :] for y in range(0, Y, 5)],
        "sagittal": [atlas_data[x, :, :] for x in range(0, X, 5)],
    }

    bold_mean = np.asarray(bold_mean)
    slices_bold = {
        "axial":    [bold_mean[:, :, z] for z in range(0, Z, 5)],
        "coronal":  [bold_mean[:, y, :] for y in range(0, Y, 5)],
        "sagittal": [bold_mean[x, :, :] for x in range(0, X, 5)],
    }

    print("Labels in Schaefer atlas:", np.unique(atlas_data).astype(int))

    # --- Create GIFs for each orientation ---
    for direction, atlas_slice_list in slices_atlas.items():
        images = []
        # Pair atlas and bold slices
        for idx, s_atlas in enumerate(atlas_slice_list):
            s_bold = slices_bold[direction][idx]

            fig, ax = plt.subplots(1, 2, figsize=(8, 4))

            # Left: Schaefer atlas labels
            cmap = cm.get_cmap("PiYG")
            cmap.set_under("black")  # for zero values
            ax[0].imshow(np.rot90(s_atlas), cmap=cmap, vmin=1, vmax=400)
            ax[0].set_title("Schaefer400")
            ax[0].axis("off")

            # Right: fMRI mean image
            ax[1].imshow(np.rot90(s_bold), cmap="PiYG")
            ax[1].set_title("BOLD (mean over time)")
            ax[1].axis("off")

            plt.suptitle(f"{direction.capitalize()} slice (every 5th)", fontsize=14)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)

            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            images.append(Image.open(buf))
            plt.close(fig)

        # Save GIF
        gif_path = f"{out_prefix}_{direction}_slices.gif"
        if images:
            images[0].save(
                fp=gif_path,
                format="GIF",
                append_images=images[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
            print(f"Saved GIF: {gif_path}")
        else:
            print(f"No slices generated for direction: {direction}")


def get_missing_schaefer_labels(atlas_path, bold_path):
    """Return expected, present, and missing Schaefer labels for one BOLD run."""
    atlas_img = nib.load(atlas_path)
    bold_img = nib.load(bold_path)

    atlas_data = atlas_img.get_fdata()
    expected = np.unique(atlas_data)
    expected = expected[expected > 0].astype(int)

    atlas_res = resample_to_img(atlas_img, bold_img, interpolation="nearest")
    res_data = atlas_res.get_fdata()
    present = np.unique(res_data)
    present = present[present > 0].astype(int)

    missing = sorted(set(expected) - set(present))
    return expected, present, missing, atlas_res

def make_gifs_missing_schaefer_vs_bold(bold_path, atlas_path=None, out_prefix="missing_schaefer"):
    """
    Visualize where the *missing* Schaefer labels would be, relative to the BOLD image.
    Creates axial/coronal/sagittal GIFs of missing-label voxels overlaid on BOLD.
    """
    if atlas_path is None:
        atlas_path = ATLAS_PATHS["Schaefer400"]

    # 1) Get missing labels and resampled atlas in BOLD space
    expected, present, missing, atlas_res = get_missing_schaefer_labels(atlas_path, bold_path)
    print(f"Expected non-zero labels: {len(expected)}")
    print(f"Present non-zero labels : {len(present)}")
    print(f"Missing labels          : {missing}")

    if not missing:
        print("No missing labels for this run – nothing to visualize.")
        return

    # 2) Load BOLD and make mean image
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()
    if bold_data.ndim == 4:
        bold_mean = bold_data.mean(axis=3)
    else:
        bold_mean = bold_data

    # 3) Resampled atlas data & missing-label mask
    atlas_data = atlas_res.get_fdata()
    X, Y, Z = atlas_data.shape
    missing_mask = np.isin(atlas_data, missing)  # True where label is missing

    # 4) Slice lists
    slices_bold = {
        "axial":    [bold_mean[:, :, z] for z in range(0, Z, 5)],
        "coronal":  [bold_mean[:, y, :] for y in range(0, Y, 5)],
        "sagittal": [bold_mean[x, :, :] for x in range(0, X, 5)],
    }
    slices_missing = {
        "axial":    [missing_mask[:, :, z] for z in range(0, Z, 5)],
        "coronal":  [missing_mask[:, y, :] for y in range(0, Y, 5)],
        "sagittal": [missing_mask[x, :, :] for x in range(0, X, 5)],
    }

    bold_vmin, bold_vmax = np.percentile(bold_mean[np.isfinite(bold_mean)], [2, 98])

    for direction, bold_slice_list in slices_bold.items():
        images = []
        for idx, s_bold in enumerate(bold_slice_list):
            s_miss = slices_missing[direction][idx]

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(np.rot90(s_bold), cmap="gray", vmin=bold_vmin, vmax=bold_vmax)
            # overlay missing labels in red
            ax.imshow(np.rot90(s_miss), cmap="Reds", alpha=0.6)
            ax.set_title(f"{direction.capitalize()} (missing labels in red)")
            ax.axis("off")

            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            images.append(Image.open(buf))
            plt.close(fig)

        if not images:
            continue

        gif_path = f"{out_prefix}_{direction}_slices.gif"
        images[0].save(
            fp=gif_path,
            format="GIF",
            append_images=images[1:],
            save_all=True,
            duration=500,
            loop=0,
        )
        print(f"Saved GIF: {gif_path}")

def show_missing_label_mask(atlas_path, bold_path, missing_labels):
    atlas_img = nib.load(atlas_path)
    bold_img = nib.load(bold_path)

    atlas_res = resample_to_img(atlas_img, bold_img, interpolation="nearest")
    atlas_res_data = atlas_res.get_fdata()

    missing_mask = np.isin(atlas_res_data, missing_labels)

    # quick axial slice visualization
    Z = atlas_res_data.shape[2]
    mid_slices = range(0, Z, 5)

    # create gif of missing label mask slices
    # Create images for each atlas and vstack them
    images = []
    for s in mid_slices:
        slice_img = missing_mask[:, :, s]

        plt.figure(figsize=(5, 5))
        plt.imshow(np.rot90(slice_img), cmap="Reds")
        plt.title(f"Missing labels mask - axial slice z={s}")
        plt.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        images.append(Image.open(buf))
        plt.close()
    # Save as GIF
    gif_path = f"missing_labels_mask_slices.gif"
    images[0].save(
        fp=gif_path,
        format='GIF',
        append_images=images[1:],  # Append all images after the first
        save_all=True,             # Essential for animated GIFs
        duration=500,              # Duration of each frame in milliseconds
        loop=0                     # 0 means loop forever, 1 means loop once, etc.
    )
    print(f"Saved GIF: {gif_path}")

if __name__ == "__main__":

    pass