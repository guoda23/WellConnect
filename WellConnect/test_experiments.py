import os
from datetime import datetime
from WellConnectController import WellConnectController

#NB: this test script is linked to 'Experiment_data' folder

BASE_WEIGHTS_LIST = [
    # 0.1 grid triples
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
TARGET_ENTROPY_LIST = [0.00, 0.50, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


MAX_DISTANCES = {
    'Age_tertiary': 1,
    'EducationLevel_tertiary': 1,
    'Gender_tertiary': 1
}

ATTRIBUTES = ['Age_tertiary', 'EducationLevel_tertiary', 'Gender_tertiary']
            #   'PHQ9_Total', 'PANCRS_TotalPositive', 'PANCRS_TotalNegative', 'PANCRS_FrequencyPositive', 'PANCRS_FrequencyNegative']
SEED = 123
GROUP_SIZE = 10
NUM_GROUPS = 8
GROUP_FORMATION = "multi-trait-entropy"
ENTROPY_TOL = float('inf')
TRAITS_OF_INTEREST =  ['Age_tertiary', 'EducationLevel_tertiary', 'Gender_tertiary']

params = {
    'max_distances': MAX_DISTANCES,
    'attributes': ATTRIBUTES,
    'seed': SEED,
    'group_size': GROUP_SIZE,
    'num_groups': NUM_GROUPS,
    'group_formation': GROUP_FORMATION,
    'entropy_tol': ENTROPY_TOL,
    'traits_of_interest': TRAITS_OF_INTEREST
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
        tolerance=ENTROPY_TOL,
        traits=ATTRIBUTES,
        seed=SEED,
        num_groups=NUM_GROUPS
    )

    for base_weights in BASE_WEIGHTS_LIST:
        recovered_weights_df = controller.run_on_groups(
            groups=groups,
            weights=base_weights
        )

        measure_dict = controller.statistical_power_analysis(
            traits_of_interest=TRAITS_OF_INTEREST, 
            recovered_weights_df=recovered_weights_df,
            weights=base_weights
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




