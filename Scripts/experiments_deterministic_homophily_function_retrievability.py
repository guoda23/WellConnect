# ───────────────────────────────
# Imports & Paths
# ───────────────────────────────
import sys
import json, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from fractions import Fraction

SCRIPT_DIR = Path(__file__).resolve().parent       # where this script is located (Scripts/)
ROOT_DIR = SCRIPT_DIR.parent                       # project root (where WellConnectController.py lives)
CONFIG_PATH = ROOT_DIR / "Config_files" / "config_deterministic_homophily_f_retrievability.json"
EXPERIMENTS_DIR = ROOT_DIR / "Experiments"
DATA_PATH = ROOT_DIR / "data" / "preprocessed.csv" # where input data is located

# add project root to sys.path so we can import WellConnectController
sys.path.append(str(ROOT_DIR))

from WellConnectController import WellConnectController

# ───────────────────────────────
# Load Config file & Unpack Variables
# ───────────────────────────────

# get config file
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

# helper function to parse fractions in  e.g. "1/3"
def parse_value(v):
    if isinstance(v, str) and "/" in v:  
        return float(Fraction(v))
    return v

# Metadata
EXPERIMENT_NAME   = cfg["experiment_name"]   
VARIANT           = cfg["variant"] #deterministic or stochastic

# Independent variables
BASE_WEIGHTS_LIST = [ #parse fractions if any
    {k: parse_value(v) for k, v in d.items()}
    for d in cfg["base_weights_list"]
]
TARGET_ENTROPY_LIST = cfg["target_entropy_list"]

# Controlled variables
MAX_DISTANCES = cfg["max_distances"]
ATTRIBUTES = cfg["attributes"]
SEEDS = cfg["seeds"]
GROUP_SIZE = cfg["group_size"]
NUM_GROUPS = cfg["num_groups"]  #number of groups to form in one cohort
GROUP_FORMATION = cfg["group_formation"]
ENTROPY_TOL = float("inf") if str(cfg["entropy_tol"]).lower() in ("inf", "infinity") else cfg["entropy_tol"]
TRAITS_OF_INTEREST = cfg["traits_of_interest"]
HOMOPHILY_FUNCTION_NAME = cfg["homophily_function"]
# regression related params
REGRESSION_TYPE = cfg["regression_type"] # "constrained" or "unconstrained"
DROP_LAST_VAR = cfg["drop_last_var"] # bool: drop last variable in regression (to prevent collinearity) and reconstruct its weight
DROP_VAR = cfg["drop_var"] # str: to drop a specific variable and reconstruct its weight (to prevent collinearity)
NAN_PENALTY = cfg["nan_penalty"]
ABNORMALITY_PENALTY = cfg["anomaly_penalty"]


# ───────────────────────────────
# Prepare Base Params Dict
# ───────────────────────────────
base_params = {
    'max_distances': MAX_DISTANCES,
    'attributes': ATTRIBUTES,
    'group_size': GROUP_SIZE,
    'num_groups': NUM_GROUPS,
    'group_formation': GROUP_FORMATION,
    'entropy_tol': ENTROPY_TOL,
    'traits_of_interest': TRAITS_OF_INTEREST,
    'homophily_function': HOMOPHILY_FUNCTION_NAME,
    'regression_type': REGRESSION_TYPE,
    'drop_last_var': DROP_LAST_VAR,
    'drop_var': DROP_VAR,
    'NaN_penalty': NAN_PENALTY,
    'Abnormality_penalty': ABNORMALITY_PENALTY,
}


# ───────────────────────────────
# Output Folder Structure
# ───────────────────────────────

# create path like: Experiments/<experiment_name>/<variant>/batch_<timestamp>/
chapter_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME / VARIANT
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_dir = chapter_dir / f"batch_{timestamp}"
batch_dir.mkdir(parents=True, exist_ok=True)

# copy config into the batch for reproducibility
shutil.copy2(CONFIG_PATH, batch_dir / "config_used.json")


# ───────────────────────────────────────────
# Experiment loop
# ───────────────────────────────────────────

# loop through each seed
for seed in SEEDS:

    print(f"\n=== Running seed {seed} ===")

    seed_dir = batch_dir / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)

    #loop through each set of base weights and entropies

    experiment_count = 1

    for target_entropy in TARGET_ENTROPY_LIST:
        controller = WellConnectController(data_path='data/preprocessed.csv',
                                            group_size=GROUP_SIZE,
                                            attributes=ATTRIBUTES,
                                            max_distances=MAX_DISTANCES)
    
        #run one cohort (i.e. fixed group formation for a given target entropy)
        groups = controller.create_groups(
            strategy=GROUP_FORMATION,
            target_entropy=target_entropy,
            traits=ATTRIBUTES,
            seed=seed,
            num_groups=NUM_GROUPS
        )

        for base_weights in BASE_WEIGHTS_LIST:
            recovered_weights_df = controller.run_on_groups(
                groups=groups,
                weights=base_weights,
                homophily_function_name=HOMOPHILY_FUNCTION_NAME,
                drop_last_var=DROP_LAST_VAR,
                drop_var=DROP_VAR,
                regression_type=REGRESSION_TYPE
            )

            measure_dict = controller.statistical_power_analysis(
                traits_of_interest=TRAITS_OF_INTEREST, 
                recovered_weights_df=recovered_weights_df,
                weights=base_weights,
                nan_penalty=NAN_PENALTY,
                anomaly_penalty=ABNORMALITY_PENALTY
            )

            params = dict(base_params) #copy base params
            #add run-specific params
            params['seed'] = seed
            params['target_entropy'] = target_entropy
            params['base_weights'] = base_weights

            run_dir = seed_dir / f"experiment_run_{experiment_count}"
            run_dir.mkdir(exist_ok=True)

            controller.save_experiment_data(groups,
                                            recovered_weights_df,
                                            params,
                                            experiment_folder=run_dir,
                                            measure_dict=measure_dict)
        

            print(f"✔ Run {experiment_count} | seed={seed}, entropy={target_entropy}, weights={base_weights}")
            experiment_count += 1




