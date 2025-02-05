# import os
# from datetime import datetime
# from WellConnectController import WellConnectController


# BASE_WEIGHTS_LIST = [
#     {'age': 0.33, 'education': 0.33, 'gender': 0.34}, # Equal distribution -> High entropy
#     {'age': 0.5, 'education': 0.2, 'gender': 0.3},    # More uneven distribution -> Lower entropy
#     {'age': 0.8, 'education': 0.1, 'gender': 0.1},    # Highly skewed distribution -> Very low entropy
#     {'age': 0.25, 'education': 0.25, 'gender': 0.5},  # Slight skew -> Moderate entropy
#     {'age': 0.6, 'education': 0.1, 'gender': 0.3},    # Another skewed distribution -> Lower entropy
#     {'age': 0.1, 'education': 0.7, 'gender': 0.2},    # Skewed with very low values for 'age' -> Lower entropy
#     {'age': 0.15, 'education': 0.35, 'gender': 0.5},  # More moderate distribution -> Moderate entropy
#     {'age': 0.1, 'education': 0.1, 'gender': 0.8},    # Highly concentrated -> Very low entropy
#     {'age': 0.45, 'education': 0.45, 'gender': 0.1},  # Skewed with high value on 'age' and 'education' -> Low entropy
#     {'age': 0.25, 'education': 0.5, 'gender': 0.25},  # Balanced distribution -> Higher entropy
# ]

# TARGET_ENTROPY_LIST = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# MAX_DISTANCES = {
#     'age': 1,
#     'education': 1,
#     'gender': 1
# }

# ATTRIBUTES = ['age', 'education', 'gender']
# SEED = 123
# GROUP_SIZE = 10
# NUM_GROUPS = 8
# GROUP_FORMATION = "multi-trait-entropy"
# ENTROPY_TOL = 0
# TRAIT_OF_INTEREST = 'gender'

# params = {
#     'max_distances': MAX_DISTANCES,
#     'attributes': ATTRIBUTES,
#     'seed': SEED,
#     'group_size': GROUP_SIZE,
#     'num_groups': NUM_GROUPS,
#     'group_formation': GROUP_FORMATION,
#     'entropy_tol': ENTROPY_TOL,
#     'trait_of_interest': TRAIT_OF_INTEREST
# }

# base_dir = "Experiment_data"
# batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# shared_folder = os.path.join(base_dir, f"batch_{batch_timestamp}")
# os.makedirs(shared_folder, exist_ok=True)


# #loop through each set of base weights and entropies
# experiment_count = 1

# for base_weights in BASE_WEIGHTS_LIST:
#     for target_entropy in TARGET_ENTROPY_LIST:
#         params['target_entropy'] = target_entropy
#         params['base_weights'] = base_weights

#         controller = WellConnectController(data_path='binary_age_gender_edu.csv',
#                                         group_size=GROUP_SIZE,
#                                         attributes=ATTRIBUTES,
#                                         max_distances=MAX_DISTANCES,
#                                         weights=base_weights)
        
#         #run one cohort 
#         groups, recovered_weights_df = controller.run(strategy=GROUP_FORMATION, target_entropy=params['target_entropy'],
#                                                     tolerance=ENTROPY_TOL,
#                                                     traits=ATTRIBUTES,
#                                                     seed=SEED,
#                                                     num_groups=NUM_GROUPS)
        
#         measure_dict = controller.statistical_power_analysis(trait_of_interest=TRAIT_OF_INTEREST, 
#                                                             recovered_weights_df=recovered_weights_df)

#         experiment_folder = os.path.join(shared_folder, f"experiment_run_{experiment_count}")
#         os.makedirs(experiment_folder, exist_ok=True)

#         controller.save_experiment_data(groups,
#                                         recovered_weights_df,
#                                         params,
#                                         experiment_folder=experiment_folder,
#                                         measure_dict=measure_dict)
        
        
#         print(f"Experiment {experiment_count} with base weights {base_weights} has been completed and saved to {experiment_folder}")

#         experiment_count += 1



#Plotting
from OutputGenerator import OutputGenerator

output_gen = OutputGenerator("Experiment_data/batch_2025-02-05_14-15-34")
data = output_gen.extract_metrics(stat_power_measure="absolute_error")
output_gen.plot_3d(data)
