# ───────────────────────────────
# Imports & Paths
# ───────────────────────────────
import sys
import json, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from fractions import Fraction
from scipy.stats import entropy


SCRIPT_DIR = Path(__file__).resolve().parent       # where this script is located (Scripts/)
ROOT_DIR = SCRIPT_DIR.parent.parent                # project root (where WellConnectController.py lives)
CONFIG_PATH = ROOT_DIR / "Config_files" / "transmission" / "config_transmission.json"
EXPERIMENTS_DIR = ROOT_DIR / "Experiments"
DATA_PATH = ROOT_DIR / "data" / "preprocessed.csv" # where input data is located

# add project root to sys.path so we can import WellConnectController
sys.path.append(str(ROOT_DIR))

from WellConnectController import WellConnectController

# ───────────────────────────────
# Helper Functions
# ───────────────────────────────

def calculate_entropy(weight_dict): # how evenly distributed are the weights? More uniform weight dist -> higher entropy
    '''pass a dictionary of values, returns the entropy'''
    weights = list(weight_dict.values())
    shannon_entropy = entropy(weights, base=2)
    return shannon_entropy

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

# Independent variables
BASE_WEIGHTS_LIST = [ #parse fractions if any
    {k: parse_value(v) for k, v in d.items()}
    for d in cfg["base_weights_list"]
]
TARGET_ENTROPY_LIST = cfg["target_entropy_list"]

# Controlled variables
MAX_DISTANCES = cfg["max_distances"]
ATTRIBUTES = cfg["attributes"]
SEEDS_GROUP_FORMATION = cfg["seeds_group_formation"]  #list of random seeds for group formation
SEEDS_TRANSMISSION = cfg["seeds_transmission"]  #list of random seeds for transmission
GROUP_SIZE = cfg["group_size"]
NUM_GROUPS = cfg["num_groups"]  #number of groups to form in one cohort
GROUP_FORMATION = cfg["group_formation"]
HOMOPHILY_FUNCTION_NAME = cfg["homophily_function"]
NOISE_STDS = cfg["noise_stds"]
MODEL_STEPS = cfg["model_steps"]
MODEL_TYPE = cfg.get("model_type")


# ───────────────────────────────
# Prepare Base Params Dict
# ───────────────────────────────
base_params = {
    'max_distances': MAX_DISTANCES,
    'attributes': ATTRIBUTES,
    'group_size': GROUP_SIZE,
    'num_groups': NUM_GROUPS,
    'group_formation': GROUP_FORMATION,
    'homophily_function': HOMOPHILY_FUNCTION_NAME,
    'model_steps': MODEL_STEPS,
    'model_type': MODEL_TYPE,
    'seeds_transmission': SEEDS_TRANSMISSION
}


# ───────────────────────────────
# Output Folder Structure
# ───────────────────────────────

# create path like: Experiments/<experiment_name>/batch_<timestamp>/
chapter_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_dir = chapter_dir / f"batch_{timestamp}"
batch_dir.mkdir(parents=True, exist_ok=True)

# copy config into the batch for reproducibility
shutil.copy2(CONFIG_PATH, batch_dir / "config_used.json")


# ───────────────────────────────────────────
# Experiment loop
# ───────────────────────────────────────────

# loop through each seed
for seed in SEEDS_GROUP_FORMATION:

    print(f"\n=== Running seed {seed} ===")

    seed_dir = batch_dir / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)

    #loop through each set of base weights and entropies
    for target_entropy in TARGET_ENTROPY_LIST:
        controller = WellConnectController(data_path=DATA_PATH,
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

        #reuse the same groups for different weights & noise levels
        for noise_std in NOISE_STDS:
            noise_dir = seed_dir / f"noise_{noise_std}"
            noise_dir.mkdir(exist_ok=True)
            
            experiment_count = 1
            
            for base_weights in BASE_WEIGHTS_LIST:
                groups = controller.predict_group_connections(
                    groups=groups,
                    weights=base_weights,
                    homophily_function_name=HOMOPHILY_FUNCTION_NAME,
                    noise_std=noise_std
                )

                contagion_histories, transition_logs = controller.simulate_depression_dynamics(groups, seeds=SEEDS_TRANSMISSION, steps=MODEL_STEPS, model_type=MODEL_TYPE)

                params = dict(base_params) #copy base params
                #add run-specific params
                params['seed'] = seed
                params['target_entropy'] = target_entropy
                params['base_weights'] = base_weights
                params['noise_std'] = noise_std

                weight_entropy = calculate_entropy(base_weights)
                run_dir = noise_dir / f"experiment_run_{experiment_count}_target_e_{target_entropy}_weight_e_{weight_entropy:.4f}"
                run_dir.mkdir(exist_ok=True)

                controller.save_experiment_data(groups,
                                                params,
                                                experiment_folder=run_dir,
                                                contagion_histories=contagion_histories,
                                                transition_logs=transition_logs
                                                )
            

                print(f"✔ Run {experiment_count} | seed={seed}, entropy={target_entropy}, "
                    f"weights={base_weights}, noise_std={noise_std}")
                experiment_count += 1


# NB: the loops are set up such that each experiment_run_{} folder each has a fixed weight but different trait entropies (so multiple pkl files)
# This is for efficiency so that we can reuse created groups for different noise levels and weights


