import os
from datetime import datetime
from WellConnectController import WellConnectController

#NB: this test script is linked to 'Experiment_data' folder

BASE_WEIGHTS_LIST = [
    # Equal distribution
    {'Age_tertiary': 1/3, 'EducationLevel_tertiary': 1/3, 'Gender_tertiary': 1/3},  # High entropy (only 1)

    # (0.50, 0.20, 0.30) -> 6 permutations
    {'Age_tertiary': 0.50, 'EducationLevel_tertiary': 0.20, 'Gender_tertiary': 0.30},
    {'Age_tertiary': 0.50, 'EducationLevel_tertiary': 0.30, 'Gender_tertiary': 0.20},
    {'Age_tertiary': 0.20, 'EducationLevel_tertiary': 0.50, 'Gender_tertiary': 0.30},
    {'Age_tertiary': 0.20, 'EducationLevel_tertiary': 0.30, 'Gender_tertiary': 0.50},
    {'Age_tertiary': 0.30, 'EducationLevel_tertiary': 0.20, 'Gender_tertiary': 0.50},
    {'Age_tertiary': 0.30, 'EducationLevel_tertiary': 0.50, 'Gender_tertiary': 0.20},

    # (0.70, 0.10, 0.20) -> 6 permutations
    {'Age_tertiary': 0.70, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.20},
    {'Age_tertiary': 0.70, 'EducationLevel_tertiary': 0.20, 'Gender_tertiary': 0.10},
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.70, 'Gender_tertiary': 0.20},
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.20, 'Gender_tertiary': 0.70},
    {'Age_tertiary': 0.20, 'EducationLevel_tertiary': 0.70, 'Gender_tertiary': 0.10},
    {'Age_tertiary': 0.20, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.70},

    # (0.80, 0.10, 0.10) -> 3 permutations (two values equal)
    {'Age_tertiary': 0.80, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.10},
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.80, 'Gender_tertiary': 0.10},
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.80},

    # (0.25, 0.25, 0.50) -> 3 permutations (two values equal)
    {'Age_tertiary': 0.25, 'EducationLevel_tertiary': 0.25, 'Gender_tertiary': 0.50},
    {'Age_tertiary': 0.25, 'EducationLevel_tertiary': 0.50, 'Gender_tertiary': 0.25},
    {'Age_tertiary': 0.50, 'EducationLevel_tertiary': 0.25, 'Gender_tertiary': 0.25},

    # (0.60, 0.10, 0.30) -> 6 permutations
    {'Age_tertiary': 0.60, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.30},
    {'Age_tertiary': 0.60, 'EducationLevel_tertiary': 0.30, 'Gender_tertiary': 0.10},
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.60, 'Gender_tertiary': 0.30},
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.30, 'Gender_tertiary': 0.60},
    {'Age_tertiary': 0.30, 'EducationLevel_tertiary': 0.60, 'Gender_tertiary': 0.10},
    {'Age_tertiary': 0.30, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.60},

    # (0.10, 0.70, 0.20) -> 6 permutations
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.70, 'Gender_tertiary': 0.20},
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.20, 'Gender_tertiary': 0.70},
    {'Age_tertiary': 0.70, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.20},
    {'Age_tertiary': 0.70, 'EducationLevel_tertiary': 0.20, 'Gender_tertiary': 0.10},
    {'Age_tertiary': 0.20, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.70},
    {'Age_tertiary': 0.20, 'EducationLevel_tertiary': 0.70, 'Gender_tertiary': 0.10},

    # (0.15, 0.35, 0.50) -> 6 permutations
    {'Age_tertiary': 0.15, 'EducationLevel_tertiary': 0.35, 'Gender_tertiary': 0.50},
    {'Age_tertiary': 0.15, 'EducationLevel_tertiary': 0.50, 'Gender_tertiary': 0.35},
    {'Age_tertiary': 0.35, 'EducationLevel_tertiary': 0.15, 'Gender_tertiary': 0.50},
    {'Age_tertiary': 0.35, 'EducationLevel_tertiary': 0.50, 'Gender_tertiary': 0.15},
    {'Age_tertiary': 0.50, 'EducationLevel_tertiary': 0.15, 'Gender_tertiary': 0.35},
    {'Age_tertiary': 0.50, 'EducationLevel_tertiary': 0.35, 'Gender_tertiary': 0.15},

    # (0.45, 0.45, 0.10) -> 3 permutations (two values equal)
    {'Age_tertiary': 0.45, 'EducationLevel_tertiary': 0.45, 'Gender_tertiary': 0.10},
    {'Age_tertiary': 0.45, 'EducationLevel_tertiary': 0.10, 'Gender_tertiary': 0.45},
    {'Age_tertiary': 0.10, 'EducationLevel_tertiary': 0.45, 'Gender_tertiary': 0.45},

    # (0.25, 0.50, 0.25) -> 3 permutations (two values equal)
    {'Age_tertiary': 0.25, 'EducationLevel_tertiary': 0.50, 'Gender_tertiary': 0.25},
    {'Age_tertiary': 0.25, 'EducationLevel_tertiary': 0.25, 'Gender_tertiary': 0.50},
    {'Age_tertiary': 0.50, 'EducationLevel_tertiary': 0.25, 'Gender_tertiary': 0.25},
]
           


TARGET_ENTROPY_LIST = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

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
ENTROPY_TOL = 0
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

base_dir = "Experiment_data"
batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
shared_folder = os.path.join(base_dir, f"batch_{batch_timestamp}")
os.makedirs(shared_folder, exist_ok=True)


#loop through each set of base weights and entropies
experiment_count = 1

for base_weights in BASE_WEIGHTS_LIST:
    for target_entropy in TARGET_ENTROPY_LIST:
        params['target_entropy'] = target_entropy
        params['base_weights'] = base_weights

        controller = WellConnectController(data_path='data/preprocessed.csv',
                                        group_size=GROUP_SIZE,
                                        attributes=ATTRIBUTES,
                                        max_distances=MAX_DISTANCES,
                                        weights=base_weights)
        
        #run one cohort 
        groups, recovered_weights_df = controller.run(strategy=GROUP_FORMATION, target_entropy=params['target_entropy'],
                                                    tolerance=ENTROPY_TOL,
                                                    traits=ATTRIBUTES,
                                                    seed=SEED,
                                                    num_groups=NUM_GROUPS)
        
        measure_dict = controller.statistical_power_analysis(traits_of_interest=TRAITS_OF_INTEREST, 
                                                            recovered_weights_df=recovered_weights_df)

        experiment_folder = os.path.join(shared_folder, f"experiment_run_{experiment_count}")
        os.makedirs(experiment_folder, exist_ok=True)

        controller.save_experiment_data(groups,
                                        recovered_weights_df,
                                        params,
                                        experiment_folder=experiment_folder,
                                        measure_dict=measure_dict)
        
        
        print(f"Experiment {experiment_count} with base weights {base_weights} has been completed and saved to {experiment_folder}")

        experiment_count += 1




