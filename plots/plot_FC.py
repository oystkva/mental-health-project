import os, sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = "/cluster/home/oystkva/project/code"
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from config import (
    PLOT_DIR,
)

from data_loader import (
    list_networks,
)
from functional_connectivity import (
    calculate_zFC_network,
    fisher_z2r,
)

##### custom cmap #####
from matplotlib.colors import LinearSegmentedColormap
clist = [(0.1, 0.6, 1.0), (0.0, 0.0, 0.0), (1.0, 0.6, 0.1)]
cmap = LinearSegmentedColormap.from_list("cmap_name", clist, N=201)
#######################


def plot_FC(zFC: np.ndarray, title: str):
    """
    Plot the functional connectivity (FC) matrix.
    Args:
        zFC (np.ndarray): 2D zFC matrix.
        title (str): Title for the plot.
    """
    rFC = fisher_z2r(zFC)
    plt.figure(figsize=(16, 12))
    plt.imshow(rFC, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title(title)
    plt.xlabel('Brain Regions')
    plt.ylabel('Brain Regions')
    if os.path.exists(os.path.join(PLOT_DIR, 'FC_matrices')) is False:
        os.makedirs(os.path.join(PLOT_DIR, 'FC_matrices'))
    plt.savefig(os.path.join(PLOT_DIR, 'FC_matrices', f"{title.replace(' ', '_').lower()}_fc_matrix.svg"), format="svg")
    plt.close()

def plot_FC_comparison(zFCs: list, titles: list, filename: str = None, format: str = "svg", p_val_matrix: list = None, networks: bool = False):
    """
    Plot multiple FC matrices for comparison.
    Args:
        zFCs (list): List of 2D zFC matrices.
        filename (str): Filename for saving the plot.
        titles (list): List of titles for each plot.
    """    
    n = len(zFCs)
    fig, axs = plt.subplots(n//2, 2, figsize=(10 * n//2, 10 * 2))

    for i in range(n):
        ax = axs[i//2][i%2] if n > 1 else axs
        
        rFC = fisher_z2r(zFCs[i])
        im = ax.imshow(rFC, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
        
        ax.set_title(titles[i])
        ax.set_xlabel('Brain Regions')
        ax.set_ylabel('Brain Regions')
        # if p_val_matrix is not None:
        #     for row in range(p_val_matrix[i].shape[0]):
        #         for col in range(p_val_matrix[i].shape[1]):
                    # if p_val_matrix[i][row, col] < 0.0001:
                    #     ax.text(col, row, '***', ha='center', va='center', color='white', fontsize=12)
                    # elif p_val_matrix[i][row, col] < 0.001:
                    #     ax.text(col, row, '**', ha='center', va='center', color='white', fontsize=12)
                    # elif p_val_matrix[i][row, col] < 0.01:
                    #     ax.text(col, row, '*', ha='center', va='center', color='white', fontsize=12)
        if networks:
            labels = list(list_networks().keys())
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
    fig.tight_layout()

    fig.colorbar(im, ax=axs.ravel().tolist(), label='Correlation Coefficient')
    # fig.text(0.5, 0.02,
    #          '*' + r'$\mathit{p} < 0.01$' ', **' + r'$\mathit{p} < 0.001$' + ', ***' + r'$\mathit{p} < 0.0001$',
    #          fontsize=20, color='black', ha='center'
    #     )
    if filename is not None:
        if networks:
            if os.path.exists(os.path.join(PLOT_DIR, 'FC_network_band_comparison')) is False:
                os.makedirs(os.path.join(PLOT_DIR, 'FC_network_band_comparison'))
            plt.savefig(os.path.join(PLOT_DIR, 'FC_network_band_comparison', f"{filename}.{format}"), format=format)
        else:
            if os.path.exists(os.path.join(PLOT_DIR, 'FC_parcel_band_comparison')) is False:
                os.makedirs(os.path.join(PLOT_DIR, 'FC_parcel_band_comparison'))
            plt.savefig(os.path.join(PLOT_DIR, 'FC_parcel_band_comparison', f"{filename}.{format}"), format=format)
        plt.close()
    else:
        plt.show()


def plot_FC_imfs(saved_data: np.ndarray):
    """
    Plot FC matrices for all IMFs in the saved data.
    Args:
        saved_data (np.ndarray): 3D array where each slice along the first axis is an IMF's fMRI data.
    """
    num_imfs = saved_data.shape[0]
    if num_imfs%3 == 0:
        rows, cols = 3, num_imfs//3
    elif num_imfs%2 == 0:
        rows, cols = 2, num_imfs//2
    else:
        rows, cols = 1, num_imfs

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows + 1))
    for i in range(rows):
        for j in range(cols):    
            zFC, _ = calculate_zFC_network(saved_data[i * cols + j])
            rFC = fisher_z2r(zFC)
            ax = axs[i][j] if num_imfs > 1 else axs
            im = ax.imshow(rFC, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
            y = 1 if i == 0 else -0.1
            ax.set_title(f'IMF {i * cols + j + 1} FC Matrix', y=y)
            if i == 0:
                ax.xaxis.set_visible(False)
            if j != 0:
                ax.yaxis.set_visible(False)
    fig.tight_layout()
    if os.path.exists(os.path.join(PLOT_DIR, 'FC_imfs')) is False:
        os.makedirs(os.path.join(PLOT_DIR, 'FC_imfs'))
    plt.savefig(os.path.join(PLOT_DIR, 'FC_imfs', f"imf_fc_matrices.svg"), format="svg")
    plt.close()