import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from project_root import PROJECT_ROOT

######################## CONFIG PARAMETERS #########################

N_CPUs = 1
TR = 0.8

###################### CONFIG PATHS #########################


FMRI_DIR = "/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4" # Only on IDUN
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BOLD_DIR = os.path.join(DATA_DIR, "BOLD_signals_parcelled")
PHENO_PATH = os.path.join(DATA_DIR, "phenotype", "demos.csv")
SUBJ_LST_DIR = os.path.join(DATA_DIR, "subject_lists")
HC_LIST = os.path.join(SUBJ_LST_DIR, "hc_subjects.txt")
MDD_LIST = os.path.join(SUBJ_LST_DIR, "mdd_subjects.txt")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")