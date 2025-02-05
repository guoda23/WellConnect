import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy
import numpy as np


class OutputGenerator:
    def __init__(self, batch_folder):
        self.batch_folder = batch_folder
        self.experiment_data = self._load_experiment_data()


    def _load_experiment_data(self):
        experiments = {}

        for folder in os.listdir(self.batch_folder): #loop over subfolders in the batch
            folder_path = os.path.join(self.batch_folder, folder)
            if os.path.isdir(folder_path): #ensure a subfolder
                print(f"Found folder: {folder}") 

                #load experiment data from pickle file
                for file in os.listdir(folder_path):
                    if file.endswith(".pkl"):
                        filepath = os.path.join(folder_path, file)
                        print(f"Loading data from: {filepath}")
                        with open(filepath, "rb") as f:
                            experiment_data = pickle.load(f)
                            experiments[filepath] = experiment_data
        
        print(f"Total experiments loaded: {len(experiments)}")
        return experiments
    

    def extract_metrics(self, stat_power_measure = 'absolute_error'):
        data =[]

        for folder, experiment in self.experiment_data.items():
            #x-axis
            trait_entropy = experiment['params']['target_entropy']        

            #y-axis
            weight_dict = experiment['params']['base_weights']
            weight_entropy = self._calculate_entropy(weight_dict)

            #z-axis
            for group_id, group_absolute_error in experiment['measure_dict'][stat_power_measure].items():
                data.append({
                    "group_id": group_id,
                    "weight_entropy": weight_entropy,
                    "trait_entropy": trait_entropy,
                    "stat_power": group_absolute_error
                })

        return data


    def _calculate_entropy(self, weight_dict):
        '''pass a dictionary of values, returns the entropy'''
        weights = list(weight_dict.values())
        shannon_entropy = entropy(weights, base=2)
        return shannon_entropy


    def plot_3d(self, data, x_label="Weight Entropy", y_label="Trait Entropy", z_label="Weight absolute error"):
        """
        Create a 3D scatter plot using the extracted data.
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

    def run_experiment(self):
        data = self.extract_metrics()
        self.plot_3d(data)



