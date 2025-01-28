import os
import pickle
from datetime import datetime

from DataHandler import DataHandler
from GroupCreator import GroupCreator
from group_creation_strategies import RandomSamplingStrategy, TraitBasedStrategy, EntropyControlledSamplingStrategy
from ConnectionPredictor import ConnectionPredictor
from RegressionRunner import RegressionRunner
from StatisticalPowerCalculator import StatisticalPowerCalculator


class WellConnectController:
    def __init__(self, data_path, group_size, attributes, max_distances, weights, file_type = 'csv'):
        self.data_handler = DataHandler(data_path, file_type=file_type)

        self.population_data = self.data_handler.read_data()
        self.group_size = group_size
        self.attributes = attributes #TODO: automate this
        self.max_distances = max_distances #TODO" automate this
        self.weights = weights 
        self.agents = self.data_handler.create_agents(self.population_data, self.attributes)
        
        
        self.connection_predictor = ConnectionPredictor(weights=self.weights, max_distances=self.max_distances)
        #modules instantiated later
        self.group_creator = None 
        self.regression_runner = None
        self.statistical_power_calculator = None


    def display_population_data(self): #TODO: remove later; testing method
        print("Population Data:")
        print(self.population_data.head()) 


    def set_group_creation_strategy(self, strategy, **kwargs):
        """Set the group creation strategy with shared and specific arguments."""
        if strategy == "random":
            self.group_creator = GroupCreator(
                RandomSamplingStrategy(self.agents, self.group_size, **kwargs)
            )
        elif strategy == "trait-based":
            self.group_creator = GroupCreator(
                TraitBasedStrategy(self.agents, self.group_size, **kwargs)
            )
        elif strategy == "entropy-controlled":
            self.group_creator = GroupCreator(
                EntropyControlledSamplingStrategy(self.agents, self.group_size, **kwargs)
            )
        else:
            raise ValueError("Unknown strategy: Choose 'random', 'trait_based', or 'entropy_controlled'")
        

    def run(self, strategy, mode = 'synthetic data', **kwargs):
        self.display_population_data() #test the data loading

        #create groups
        self.set_group_creation_strategy(strategy, **kwargs)
        groups = self.group_creator.create_groups()
        # print("Created Groups:", groups)

        for group in groups:
            group.create_group_graph() #create group graphs

            #run the social connection predictions (update graphs with weights)
            self.connection_predictor.predict_weights(group.network)
        
        for agent in groups[1].members:
            print(agent)

        #run linear regression
        self.regression_runner = RegressionRunner(attributes=self.attributes, max_distances=self.max_distances)
        recovered_weights_df = self.regression_runner.perform_group_regression(groups=groups)
        
        if mode == 'synthetic data':
            self.regression_runner.display_results(recovered_weights_df=recovered_weights_df, true_weights=self.weights)
        elif mode == 'real data':
            self.regression_runner.display_results(recovered_weights_df=recovered_weights_df, true_weights=None)

        return groups, recovered_weights_df  #TODO: put into an storage file -> save_experiment_data()


    def statistical_power_analysis(self, trait_of_interest, recovered_weights_df):
        self.statistical_power_calculator = StatisticalPowerCalculator(recovered_weights_df=recovered_weights_df, true_weights=self.weights)
        measure_dict = self.statistical_power_calculator.evaluate_predictive_power(attribute=trait_of_interest)
        return measure_dict
    

    def save_experiment_data(self, groups, recovered_weights_df, params, experiment_folder, measure_dict = None):
        os.makedirs(experiment_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(experiment_folder, f"experiment_{timestamp}.pkl")

        experiment_data = { #bundle data
            "groups": groups,
            "recovered_weights_df": recovered_weights_df,
            "params": params,
            "measure_dict": measure_dict,
            "timestamp": timestamp
        }

        
        with open(filename, "wb") as f:
            pickle.dump(experiment_data, f)

        print(f"Experiment data saved to {filename}")

