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
GROUP_FORMATION = 'entropy-based' #'random' #'biased'
TRAIT_BIAS = 'age+edu+gender' #'age+edu' #'age' # #
TOLERANCE = 0.01
TARGET_ENTROPY = 1.0
ENTROPY_TOL = 0.05
TRAIT_OF_INTEREST = 'gender'


controller = WellConnectController(data_path='binary_age_gender_edu.csv',
                                   group_size=GROUP_SIZE,
                                   attributes=ATTRIBUTES,
                                   max_distances=MAX_DISTANCES,
                                   weights=BASE_WEIGHTS)

groups, group_graphs, recovered_weights_df = controller.run("entropy_controlled", target_entropy = TARGET_ENTROPY,
               tolerance = ENTROPY_TOL,
               trait = TRAIT_OF_INTEREST,
               seed = SEED,
               num_groups = NUM_GROUPS)

measure_dict = controller.statistical_power_analysis(trait_of_interest=TRAIT_OF_INTEREST , recovered_weights_df=recovered_weights_df)
print(measure_dict)