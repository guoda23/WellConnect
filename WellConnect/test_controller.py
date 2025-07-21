import os
from datetime import datetime

from WellConnectController import WellConnectController

BASE_WEIGHTS = { # Sum of weights should be 1
    'age': 0.33,
    'education': 0.33,
    'gender': 0.34  
}

MAX_DISTANCES = {
    'age': 1,
    'education': 1,
    'gender': 1
}

ATTRIBUTES = ['age', 'education', 'gender']
SEED = 123
GROUP_SIZE = 10
NUM_GROUPS = 10
GROUP_FORMATION = 'entropy-controlled'
TARGET_ENTROPY = 0.22
ENTROPY_TOL = 0
TRAIT_OF_INTEREST = 'gender'

params = {
    'base_weights': BASE_WEIGHTS,
    'max_distances': MAX_DISTANCES,
    'attributes': ATTRIBUTES,
    'seed': SEED,
    'group_size': GROUP_SIZE,
    'num_groups': NUM_GROUPS,
    'group_formation': GROUP_FORMATION,
    'target_entropy': TARGET_ENTROPY,
    'entropy_tol': ENTROPY_TOL,
    'trait_of_interest': TRAIT_OF_INTEREST
}

base_dir = "experiments"
batch_timestemp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
shared_folder = os.path.join(base_dir, f"batch_{batch_timestemp}")
os.makedirs(shared_folder, exist_ok=True)

controller = WellConnectController(data_path='binary_age_gender_edu.csv',
                                   group_size=GROUP_SIZE,
                                   attributes=ATTRIBUTES,
                                   max_distances=MAX_DISTANCES,
                                   weights=BASE_WEIGHTS)

groups, recovered_weights_df = controller.run(strategy=GROUP_FORMATION, target_entropy = TARGET_ENTROPY,
               tolerance = ENTROPY_TOL,
               trait = TRAIT_OF_INTEREST,
               seed = SEED,
               num_groups = NUM_GROUPS)

measure_dict = controller.statistical_power_analysis(trait_of_interest=TRAIT_OF_INTEREST , recovered_weights_df=recovered_weights_df)
controller.save_experiment_data(groups,
                                recovered_weights_df,
                                params,
                                experiment_folder=shared_folder,
                                measure_dict=measure_dict)



#test plot generation
from OutputGenerator import OutputGenerator

# batch_folder = "experiments/batch_2025-01-28_13-39-20"
batch_folder = "experiments/batch_2025-04-22_10-59-18"

output_gen = OutputGenerator(batch_folder)

# Step 1: Extract the data
data = output_gen.extract_metrics(stat_power_measure="absolute_error")

# Step 2: Plot the 3D scatter plot
output_gen.plot_3d(data)
