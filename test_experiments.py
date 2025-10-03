#Deprecated - see Scripts/experiments_homophily_function_retrieval.py for the updated version

import os
from datetime import datetime
from WellConnectController import WellConnectController
import numpy as np

#NB: this test script is linked to 'Experiment_data' folder

BASE_WEIGHTS_LIST = [
    # 0.1 grid triplesf
    {'Age_tertiary': 1.0, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.9, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.9, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.8, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.8, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.8, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.7, 'EducationLevel_tertiary': 0.3, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.7, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.7, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.7, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.3},
    {'Age_tertiary': 0.6, 'EducationLevel_tertiary': 0.4, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.6, 'EducationLevel_tertiary': 0.3, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.6, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.6, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.3},
    {'Age_tertiary': 0.6, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.4},
    {'Age_tertiary': 0.5, 'EducationLevel_tertiary': 0.5, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.5, 'EducationLevel_tertiary': 0.4, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.5, 'EducationLevel_tertiary': 0.3, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.5, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.3},
    {'Age_tertiary': 0.5, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.4},
    {'Age_tertiary': 0.5, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.5},
    {'Age_tertiary': 0.4, 'EducationLevel_tertiary': 0.6, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.4, 'EducationLevel_tertiary': 0.5, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.4, 'EducationLevel_tertiary': 0.4, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.4, 'EducationLevel_tertiary': 0.3, 'Gender_tertiary': 0.3},
    {'Age_tertiary': 0.4, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.4},
    {'Age_tertiary': 0.4, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.5},
    {'Age_tertiary': 0.4, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.6},
    {'Age_tertiary': 0.3, 'EducationLevel_tertiary': 0.7, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.3, 'EducationLevel_tertiary': 0.6, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.3, 'EducationLevel_tertiary': 0.5, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.3, 'EducationLevel_tertiary': 0.4, 'Gender_tertiary': 0.3},
    {'Age_tertiary': 0.3, 'EducationLevel_tertiary': 0.3, 'Gender_tertiary': 0.4},
    {'Age_tertiary': 0.3, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.5},
    {'Age_tertiary': 0.3, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.6},
    {'Age_tertiary': 0.3, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.7},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.8, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.7, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.6, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.5, 'Gender_tertiary': 0.3},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.4, 'Gender_tertiary': 0.4},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.3, 'Gender_tertiary': 0.5},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.6},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.7},
    {'Age_tertiary': 0.2, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.8},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.9, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.8, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.7, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.6, 'Gender_tertiary': 0.3},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.5, 'Gender_tertiary': 0.4},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.4, 'Gender_tertiary': 0.5},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.3, 'Gender_tertiary': 0.6},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.7},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.8},
    {'Age_tertiary': 0.1, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 0.9},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 1.0, 'Gender_tertiary': 0.0},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.9, 'Gender_tertiary': 0.1},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.8, 'Gender_tertiary': 0.2},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.7, 'Gender_tertiary': 0.3},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.6, 'Gender_tertiary': 0.4},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.5, 'Gender_tertiary': 0.5},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.4, 'Gender_tertiary': 0.6},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.3, 'Gender_tertiary': 0.7},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.2, 'Gender_tertiary': 0.8},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.1, 'Gender_tertiary': 0.9},
    {'Age_tertiary': 0.0, 'EducationLevel_tertiary': 0.0, 'Gender_tertiary': 1.0},

    # Special equal-distribution case
    {'Age_tertiary': 1/3, 'EducationLevel_tertiary': 1/3, 'Gender_tertiary': 1/3},


    #low entropy values to supplement
    {'Age_tertiary': 0.99, 'EducationLevel_tertiary': 0.005, 'Gender_tertiary': 0.005},
    {'Age_tertiary': 0.98, 'EducationLevel_tertiary': 0.01, 'Gender_tertiary': 0.01},
    {'Age_tertiary': 0.97, 'EducationLevel_tertiary': 0.015, 'Gender_tertiary': 0.015},

    {'EducationLevel_tertiary': 0.99, 'Age_tertiary': 0.005, 'Gender_tertiary': 0.005},
    {'EducationLevel_tertiary': 0.98, 'Age_tertiary': 0.01, 'Gender_tertiary': 0.01},
    {'EducationLevel_tertiary': 0.97, 'Age_tertiary': 0.015, 'Gender_tertiary': 0.015},

    {'Gender_tertiary': 0.99, 'Age_tertiary': 0.005, 'EducationLevel_tertiary': 0.005},
    {'Gender_tertiary': 0.98, 'Age_tertiary': 0.01, 'EducationLevel_tertiary': 0.01},
    {'Gender_tertiary': 0.97, 'Age_tertiary': 0.015, 'EducationLevel_tertiary': 0.015},

]

           


# TARGET_ENTROPY_LIST = [0.00, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# TARGET_ENTROPY_LIST = [0.00, 0.1, 0.20, 0.30, 0.40, 0.50, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# TARGET_ENTROPY_LIST = [0.00, 0.50, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# TARGET_ENTROPY_LIST = np.round(np.linspace(0.0, 3.32, num=30), 2)

#all value combinations
# TARGET_ENTROPY_LIST = [0.0, 0.469, 0.7219, 0.8813, 0.9219, 0.971, 1.0, 1.1568, 1.2955, 1.3568, 1.361, 1.371, 1.4855, 1.5219, 1.571, 1.6855, 1.7219, 1.761, 1.771, 1.8464, 1.8955, 1.9219, 1.961, 1.971, 2.0464, 2.1219, 2.161, 2.171, 2.2464, 2.3219, 2.371, 2.4464, 2.5219, 2.6464, 2.7219, 2.8464, 2.9219, 3.1219, 3.3219]

#shortened list of all value combinations
TARGET_ENTROPY_LIST = [0.0, 0.469, 0.7219, 0.8813, 0.971, 1.0, 1.1568, 1.3568, 1.361, 1.371, 1.5219, 1.571, 1.6855, 1.7219, 1.771, 1.8464, 1.961, 1.971, 2.0464,2.2464, 2.4464, 2.5219, 2.9219, 3.1219, 3.3219]

MAX_DISTANCES = {
    'Age_tertiary': 1,
    'EducationLevel_tertiary': 1,
    'Gender_tertiary': 1
}

ATTRIBUTES = ['Age_tertiary', 'EducationLevel_tertiary', 'Gender_tertiary']
            #   'PHQ9_Total', 'PANCRS_TotalPositive', 'PANCRS_TotalNegative', 'PANCRS_FrequencyPositive', 'PANCRS_FrequencyNegative']
SEED = 20
GROUP_SIZE = 10
NUM_GROUPS = 8
GROUP_FORMATION = "multi-trait-entropy"
TRAITS_OF_INTEREST =  ['Age_tertiary', 'EducationLevel_tertiary', 'Gender_tertiary']
HOMOPHILY_FUNCTION_NAME = "linear_deterministic"  #or "linear_stochastic"
REGRESSION_TYPE = "constrained"  #or "unconstrained"
DROP_LAST_VAR = False # trop drop last variable in regression (to prevent collinearity) and reconstruct its weight
DROP_VAR = None #to drop a specific variable and reconstruct its weight (to prevent collinearity)
NAN_PENALTY = 1.0
ABNORMALITY_PENALTY = 1.0

params = {
    'max_distances': MAX_DISTANCES,
    'attributes': ATTRIBUTES,
    'seed': SEED,
    'group_size': GROUP_SIZE,
    'num_groups': NUM_GROUPS,
    'group_formation': GROUP_FORMATION,
    'traits_of_interest': TRAITS_OF_INTEREST,
    'homophily_function': HOMOPHILY_FUNCTION_NAME,
    #regression related params
    'regression_type': REGRESSION_TYPE,
    'drop_last_var': DROP_LAST_VAR,
    'drop_var': DROP_VAR,
    #TODO: add penalties dynamically
    'NaN_penalty': NAN_PENALTY,
    'Abnormality_penalty': ABNORMALITY_PENALTY,
}

# ───────────────────────────────────────────
# EXPERIMENT LOOP
# ───────────────────────────────────────────


base_dir = "Experiment_data"
batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
shared_folder = os.path.join(base_dir, f"batch_{batch_timestamp}")
os.makedirs(shared_folder, exist_ok=True)


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
        seed=SEED,
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

        params['target_entropy'] = target_entropy
        params['base_weights'] = base_weights

        experiment_folder = os.path.join(shared_folder, f"experiment_run_{experiment_count}")
        os.makedirs(experiment_folder, exist_ok=True)

        controller.save_experiment_data(groups,
                                        recovered_weights_df,
                                        params,
                                        experiment_folder=experiment_folder,
                                        measure_dict=measure_dict)
        
        
        print(f"Experiment {experiment_count} completed — target entropy {target_entropy}, weights {base_weights}")

        experiment_count += 1




