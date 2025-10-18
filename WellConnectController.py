import os
import pickle
from datetime import datetime

from DataHandler import DataHandler
from GroupCreator import GroupCreator
from group_creation_strategies import RandomSamplingStrategy, TraitBasedStrategy, EntropyControlledSamplingStrategy, MultiTraitEntropySamplingStrategy
from ConnectionPredictor import ConnectionPredictor
from RegressionRunner import RegressionRunner
from StatisticalPowerCalculator import StatisticalPowerCalculator
from OutputGenerator import OutputGenerator
from Visualizer3DScatterplot import Visualizer3DScatterPlot
from TransmissionSimulator import TransmissionSimulator

class WellConnectController:
    def __init__(self, data_path, group_size, attributes, max_distances, file_type = 'csv'):
        self.data_handler = DataHandler(data_path, file_type=file_type)

        self.population_data = self.data_handler.read_data()
        self.group_size = group_size
        self.attributes = attributes #TODO: automate this? maybe good to keep separate for subsetting traits
        self.max_distances = max_distances #TODO" automate this
        self.agents = self.data_handler.create_agents(self.population_data, self.attributes)
        
        #modules instantiated later
        self.connection_predictor = None
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
        elif strategy == "multi-trait-entropy":
            self.group_creator = GroupCreator(
                MultiTraitEntropySamplingStrategy(self.agents, self.group_size, **kwargs)
            )
        else:
            raise ValueError("Unknown strategy: Choose 'random', 'trait_based', or 'entropy_controlled'")
    

    def create_groups(self, strategy, **kwargs):
        self.set_group_creation_strategy(strategy, **kwargs)
        groups = self.group_creator.create_groups()
        return groups
    

    def predict_group_connections(self, groups,  homophily_function_name, weights, **kwargs):
        
        self.connection_predictor = ConnectionPredictor(weights=weights, max_distances=self.max_distances, homophily_function_name=homophily_function_name)

        for group in groups:
            group.create_group_graph() #create group graphs
            #run the social connection predictions (update graphs with weights)
            self.connection_predictor.predict_weights(group.network, **kwargs)

        return groups

    def recover_group_connections(self, groups , weights, drop_last_var, drop_var, regression_type, mode = 'synthetic data'):
        #run regression
        self.regression_runner = RegressionRunner(attributes=list(weights.keys()), max_distances=self.max_distances, regression_type=regression_type)
        recovered_weights_df = self.regression_runner.perform_group_regression(groups=groups, drop_last_var=drop_last_var, drop_var=drop_var)

        if mode == 'synthetic data':
            self.regression_runner.display_results(recovered_weights_df=recovered_weights_df, true_weights=weights)
        elif mode == 'real data':
            self.regression_runner.display_results(recovered_weights_df=recovered_weights_df, true_weights=None)

        return recovered_weights_df  #put into a storage file by save_experiment_data()


    def statistical_power_analysis(self, traits_of_interest, recovered_weights_df, weights, **kwargs):
        self.statistical_power_calculator = StatisticalPowerCalculator(recovered_weights_df=recovered_weights_df, true_weights=weights)
        measure_dict = self.statistical_power_calculator.evaluate_predictive_power_trait_specific(attributes=traits_of_interest, **kwargs)
        return measure_dict
    

    def simulate_depression_dynamics(self, groups, seed, steps, model_type="HMDaModel"):
        """
        Runs the depression transmission model (default HMDaModel) on every group
        and returns a dictionary mapping group_id -> simulation history (np.array of shape [steps, 3]).
        """
        contagion_sim = TransmissionSimulator(model_type=model_type, seed=seed)
        contagion_history_dict = {}

        for group in groups:
            history, _ = contagion_sim.run(group, steps=steps)
            contagion_history_dict[group.group_id] = history

        return contagion_history_dict
    

    def save_experiment_data(self, groups, params, experiment_folder,
                            recovered_weights_df=None, measure_dict=None,
                            contagion_histories=None):
        
        os.makedirs(experiment_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(experiment_folder, f"experiment_{timestamp}.pkl")

        experiment_data = {
            "groups": groups,
            "params": params,
            "timestamp": timestamp,
        }

        if recovered_weights_df is not None:
            experiment_data["recovered_weights_df"] = recovered_weights_df
        if measure_dict is not None:
            experiment_data["measure_dict"] = measure_dict
        if contagion_histories is not None:
            experiment_data["contagion_histories"] = contagion_histories  # {group_id: np.ndarray}

        with open(filename, "wb") as f:
            pickle.dump(experiment_data, f)

        print(f"Experiment data saved to {filename}")




