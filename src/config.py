import sys, os


######################## CONFIG PARAMETERS #########################

N_CPUs = 1
TR = 0.8

###################### CONFIG PARAMETERS IDUN SET UP #########################

PROJECT_ROOT = "/cluster/home/oystkva/project/code"

FMRI_DIR = "/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BOLD_DIR = os.path.join(DATA_DIR, "BOLD_signals_parcelled")
PHENO_PATH = os.path.join(DATA_DIR, "phenotype", "demos.csv")
SUBJ_LST_DIR = os.path.join(DATA_DIR, "subject_lists")
HC_LIST = os.path.join(SUBJ_LST_DIR, "hc_subjects.txt")
MDD_LIST = os.path.join(SUBJ_LST_DIR, "mdd_subjects.txt")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

#####################################################################################

######################## CONFIG PARAMETERS LOCAL SET UP #########################

# PROJECT_ROOT = "C:\\Users\\oykva\\OneDrive - NTNU\\Semester 9\\Prosjektoppgave\\project_code"

# FMRI_DIR = os.path.join(PROJECT_ROOT, "fmriprep-25.1.4")
# DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# PHENO_PATH = os.path.join(DATA_DIR, "phenotype", "demos.csv")
# SUBJ_LST_DIR = os.path.join(DATA_DIR, "subject_lists")
# HC_LIST = os.path.join(SUBJ_LST_DIR, "hc_subjects.txt")
# MDD_LIST = os.path.join(SUBJ_LST_DIR, "mdd_subjects.txt")