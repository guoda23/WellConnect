import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy
import numpy as np
import networkx as nx

from Visualizer3DScatterplot import Visualizer3DScatterPlot


class OutputGenerator:
    def __init__(self, batch_folder):
        self.batch_folder = batch_folder
        self.experiment_data = self._load_experiment_data()


    def _load_experiment_data(self):
        all_experiments_data = {}

        for folder in os.listdir(self.batch_folder): #loop over subfolders in the batch
            folder_path = os.path.join(self.batch_folder, folder)
            if os.path.isdir(folder_path): #ensure a subfolder (e.g. "experiment_run_1")
                # print(f"Found folder: {folder}") 

                #load experiment data from pickle file
                for file in os.listdir(folder_path):
                    if file.endswith(".pkl"):
                        filepath = os.path.join(folder_path, file)
                        # print(f"Loading data from: {filepath}")

                        with open(filepath, "rb") as f:
                            experiment_data = pickle.load(f) #load one file
                            all_experiments_data[filepath] = experiment_data #store in main dict
        
        # print(f"Total experiments loaded: {len(all_experiments_data)}")
        return all_experiments_data
    

    def extract_metrics(self, stat_power_measure = 'absolute_error'):
        data =[]

        for folder, experiment in self.experiment_data.items(): #for cohort in cohort batch
            #extracting aggregate variables for the entire cohort
            trait_entropy = experiment['params']['target_entropy'] 

            weight_dict = experiment['params']['base_weights']
            weight_entropy = self._calculate_entropy(weight_dict)

            measure_dict = experiment["measure_dict"][stat_power_measure]

            groups_list = experiment["groups"]

            recovered_weights_df = experiment["recovered_weights_df"]

            #unpacking aggregate variable for the entire cohort to represent a specific group in the cohort
            for group in groups_list:
                group_id_within_cohort = group.group_id - 1 #TODO: check if this indexing is right (now reduced by 1 bcz groups start at 1, not 0)
                group_absolute_error = measure_dict[group_id_within_cohort] 

                # Append extracted data to the list
                data.append({
                    "weight_entropy": weight_entropy,  # X-axis
                    "trait_entropy": trait_entropy,  # Y-axis
                    "stat_power": group_absolute_error,  # Z-axis
                    "group": group,  # The Group Object
                    "recovered_weights_df": recovered_weights_df, # The recovered weights DataFrame
                    "true_weights": weight_dict,  # The true weights
                    "row_of_interest_in_table": group_id_within_cohort  # The row of interest in the recovered weights table
                })

        return data


    def _calculate_entropy(self, weight_dict): # how evenly distributed are the weights? More uniform weight dist -> higher entropy
        '''pass a dictionary of values, returns the entropy'''
        weights = list(weight_dict.values())
        shannon_entropy = entropy(weights, base=2)
        return shannon_entropy


    def plot_3d(self, data, x_label="Weight Entropy", y_label="Trait Entropy", z_label="Weight absolute error"): #TODO: remove once interactive plot is ready (run_3d_visualization())
        """
        Create a 3D scatter plot using the extracted data.
        !Non-interactive!
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x_data = [d["weight_entropy"] for d in data]
        y_data = [d["trait_entropy"]  for d in data]
        z_data = [d["stat_power"] for d in data]

        print(len(x_data), len(y_data), len(z_data))
        # Scatter plot
        color_map = plt.cm.viridis 
        scatter = ax.scatter(x_data, y_data, z_data, c=z_data, cmap="viridis", marker="o")

        # Add axis labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.title("3D Visualization of Group Metrics")

        # Add a color bar to show the mapping of Z-axis values to colors
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, alpha=0.5)
        cbar.set_label(z_label)

        # Show the plot
        plt.show()


    def run_3d_visualization(self, x_label="Weight Entropy", y_label="Trait Entropy", z_label="Weight absolute error"):
        """Runs the interactive 3D plot in the browser"""
        data = self.extract_metrics()
        visualizer = Visualizer3DScatterPlot(data, x_label, y_label, z_label)
        visualizer.run()