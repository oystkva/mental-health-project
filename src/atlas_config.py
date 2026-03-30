import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROJECT_ROOT


#region list_ATLAS_networks helper functions 
def list_Yan17_networks() -> dict[str, list[int]]:
    """
    https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/MNI/kong17/400Parcels_Kong2022_17Networks_FSLMNI152_2mm.nii.gz
    """
    network_map = {k: [] for k in list_Schaefer17_networks().keys()}
    with open(os.path.join(PROJECT_ROOT, "src", "brain_atlases", "Yan2023_homotopic_400Parcels_Kong2022_17Networks_LUT.txt"), "r") as f:
        lines = f.readlines()
    lines = [network.strip().split(' ') for network in lines]
    # networks_labels = list(set([line[1].split("_")[2] for line in lines]))
    # print(networks_labels)
    for i in range(len(lines)):
        if lines[i][1].split("_")[2] not in network_map:
            network_map[lines[i][1].split("_")[2]] = []
        network_map[lines[i][1].split("_")[2]].extend([int(lines[i][0]) - 1])
    return network_map


def list_Schaefer17_networks() -> dict[str, list[int]]:
    network_map = {}
    with open(os.path.join(PROJECT_ROOT, "src", "brain_atlases", "Schaefer2018_400Parcels_Kong2022_17Networks_order.txt"), "r") as f:
        lines = f.readlines()
    lines = [network.strip().split('\t') for network in lines]
    # networks_labels = list(set([line[1].split("_")[2] for line in lines]))
    # print(networks_labels)
    for i in range(len(lines)):
        if lines[i][1].split("_")[2] not in network_map:
            network_map[lines[i][1].split("_")[2]] = []
        network_map[lines[i][1].split("_")[2]].extend([int(lines[i][0]) - 1])
    return network_map


def list_Tian3_networks() -> dict[str, list[int]]:
    """
    Labels found on: https://github.com/yetianmed/subcortex/blob/master/Group-Parcellation/3T/Cortex-Subcortex/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S2_label.txt
    """
    network_map = {
        "MedialTemporal": [], 
        "Striatal": [], 
        "Thalamic": []
    }
    with open(os.path.join(PROJECT_ROOT, "src", "brain_atlases", "Tian_Subcortex_S2_3T_info.txt"), "r") as f:
        lines = f.readlines()
    lines = [network.strip() for network in lines]
    for i in range(0, len(lines), 2):
        label = lines[i]
        if "HIP" in label.upper() or "AMY" in label.upper():
            network_map["MedialTemporal"].extend([int(lines[i+1].split(" ")[0]) - 1 + 400])
        elif "THA" in label.upper():
            network_map["Thalamic"].extend([int(lines[i+1].split(" ")[0]) - 1 + 400])
        else:
            network_map["Striatal"].extend([int(lines[i+1].split(" ")[0]) - 1 + 400]) 
    return network_map


def list_Buckner1_networks() -> dict[str, list[int]]:
    return {
        "Cerebellar": [432, 433]
    }
#endregion

def list_networks(atlas = "Yan2023") -> dict[str, list[int]]:
    network_map = {}
    if atlas == "Yan2023":
        network_map.update(list_Yan17_networks())
    else:
        network_map.update(list_Schaefer17_networks())
    network_map.update(list_Tian3_networks())
    network_map.update(list_Buckner1_networks())
    return network_map