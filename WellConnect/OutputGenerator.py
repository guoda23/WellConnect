import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy
import numpy as np
import networkx as nx
import math
from collections import defaultdict

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
    

    def extract_metrics(self, stat_power_measure = 'absolute_error', trait_of_interest='Gender', target_entropy=True):
        data =[]

        for folder, experiment in self.experiment_data.items(): #for cohort in cohort batch
            #extracting aggregate variables for the entire cohort
            trait_entropy = experiment['params']['target_entropy'] 

            weight_dict = experiment['params']['base_weights']
            weight_entropy = self._calculate_entropy(weight_dict)

            measure_dict = experiment["measure_dict"][trait_of_interest][stat_power_measure]

            groups_list = experiment["groups"]

            recovered_weights_df = experiment["recovered_weights_df"]

            #unpacking aggregate variable for the entire cohort to represent a specific group in the cohort
            for group in groups_list:
                group_id_within_cohort = group.group_id - 1 #TODO: check if this indexing is right (now reduced by 1 bcz groups start at 1, not 0)
                group_absolute_error = measure_dict[group_id_within_cohort] 

                trait_entropy_value = trait_entropy if target_entropy is True else group.real_entropy #choose between target and real entropy of the group

                # Append extracted data to the list
                data.append({
                    "weight_entropy": weight_entropy,  # X-axis
                    "trait_entropy": trait_entropy_value,  # Y-axis
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


    def plot_3d(self, trait_of_interest, x_label="Weight Entropy", y_label="Trait Entropy", z_label="Weight absolute error", target_entropy=True): #TODO: remove once interactive plot is ready (run_3d_visualization())
        """
        Create a 3D scatter plot using the extracted data.
        !Non-interactive!
        """
        data = self.extract_metrics(trait_of_interest=trait_of_interest, target_entropy=target_entropy)
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


    def plot_heatmaps(
        self, traits, x_label="Weight Entropy", y_label="Trait Entropy",
        target_entropy=True, cmap="viridis", vmin=None, vmax=None,
        annotate=False, fmt=".2f", figsize_per_plot=(5, 4),
        round_decimals=3, share_scale=True
    ):
        traits = list(traits)
        per_trait_cells = []
        all_x, all_y = set(), set()

        # Collect data for all traits
        for trait in traits:
            data = self.extract_metrics(trait_of_interest=trait, target_entropy=target_entropy)
            cell = {}
            for d in data:
                x = d["weight_entropy"]
                y = d["trait_entropy"]
                if round_decimals is not None:
                    x = round(float(x), round_decimals)
                    y = round(float(y), round_decimals)
                key = (x, y)
                cell.setdefault(key, []).append(d["stat_power"])
            per_trait_cells.append((trait, cell))
            for (x, y) in cell.keys():
                all_x.add(x); all_y.add(y)

        x_vals = sorted(all_x)
        y_vals = sorted(all_y)

        # Build grids per trait
        grids = []
        for trait, cell in per_trait_cells:
            grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
            for i, y in enumerate(y_vals):
                for j, x in enumerate(x_vals):
                    zs = cell.get((x, y))
                    if zs:
                        grid[i, j] = float(np.mean(zs))
            grids.append((trait, grid))

        # Shared color scale if desired
        if (vmin is None or vmax is None) and share_scale:
            stacked = np.concatenate([g[1].ravel() for g in grids])
            stacked = stacked[~np.isnan(stacked)]
            if stacked.size > 0:
                if vmin is None: vmin = float(np.nanmin(stacked))
                if vmax is None: vmax = float(np.nanmax(stacked))

        # Plotting
        n = len(traits)
        fig, axes = plt.subplots(
            1, n, figsize=(figsize_per_plot[0]*n + 1.5, figsize_per_plot[1]),  # Add space for colorbar
            squeeze=False,
        )
        axes = axes[0]

        ims = []
        for i, (ax, (trait, grid)) in enumerate(zip(axes, grids)):
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ims.append(im)

            ax.set_xticks(np.arange(len(x_vals)))
            ax.set_yticks(np.arange(len(y_vals)))
            ax.set_xticklabels([f"{v:.2f}" for v in x_vals], rotation=45, ha="right")
            ax.set_yticklabels([f"{v:.2f}" for v in y_vals])

            ax.set_xlabel(x_label)
            if i == 0:
                ax.set_ylabel(y_label)

            ax.set_title(f"Absolute error ({trait.replace('_', ' ')})")

            if annotate:
                for i in range(len(y_vals)):
                    for j in range(len(x_vals)):
                        val = grid[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

        # Add shared colorbar to the right
        fig.subplots_adjust(right=0.88)  # Leave space for colorbar
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # Manual placement
        cbar = fig.colorbar(ims[0], cax=cbar_ax)
        cbar.set_label("Absolute error")

        # Ensure x-axis labels are not cut off
        plt.tight_layout(rect=[0, 0, 0.88, 1])

        plt.show()
        return fig, axes


    def run_3d_visualization(self, x_label="Weight Entropy", y_label="Trait Entropy", stat_power_measure='Absolute Error', trait_of_interest='Gender', target_entropy=True):
        """Runs the interactive 3D plot in the browser"""
        #capitalize strip
        z_label = f'{stat_power_measure.strip()} for {trait_of_interest}'
        data = self.extract_metrics(trait_of_interest=trait_of_interest, target_entropy=target_entropy)
        visualizer = Visualizer3DScatterPlot(data, x_label, y_label, z_label)
        visualizer.run()



    def _weights_match(self, w_a: dict, w_b: dict, tol: float = 1e-9) -> bool:
        """Return True if two weight dicts match within tolerance (keys + values)."""
        if set(w_a.keys()) != set(w_b.keys()):
            return False
        for k in w_a.keys():
            if not math.isclose(float(w_a[k]), float(w_b[k]), rel_tol=0, abs_tol=tol):
                return False
        return True

    def _make_uniform_weights_like(self, template_weights: dict) -> dict:
        """Build a uniform weight dict over the same keys as template_weights."""
        n = len(template_weights)
        if n == 0:
            return {}
        val = 1.0 / n
        return {k: val for k in template_weights.keys()}


    def plot_real_entropy_boxplots(self, weight_choice=None, tol=1e-9, figsize=(7,4), title="Realized group entropies by target entropy (single weight choice)", savepath=None, show=True):
        """
        Make boxplots of realized group entropies (group.real_entropy) for each target_entropy,
        restricted to experiments that used a chosen weight vector.

        Parameters
        ----------
        weight_choice : dict or None
            Weight dict to filter experiments on (e.g., {'Age': 1/3, 'Gender': 1/3, 'Edu': 1/3}).
            If None, a uniform dict over the first experiment's weight keys is used.
        tol : float
            Tolerance for matching base_weights to `weight_choice`. Default 1e-9.
        figsize : tuple
            Matplotlib figure size (width, height). Default (7, 4).
        title : str or None
            Title for the plot. Use None for no title.
        savepath : str or None
            If given, saves the figure to this path (e.g. 'out/boxplots.png').
        show : bool
            If True, displays the plot with plt.show().
        """
        def weights_match(w_a, w_b):
            if set(w_a.keys()) != set(w_b.keys()):
                return False
            return all(abs(float(w_a[k]) - float(w_b[k])) <= tol for k in w_a)

        # default: uniform weights over first experiment keys
        if weight_choice is None:
            for exp in self.experiment_data.values():
                keys = list(exp['params']['base_weights'].keys())
                if keys:
                    val = 1.0 / len(keys)
                    weight_choice = {k: val for k in keys}
                    break

        buckets = defaultdict(list)

        for exp in self.experiment_data.values():
            base_w = exp['params']['base_weights']
            if not weights_match(base_w, weight_choice):
                continue

            target_ent = float(exp['params']['target_entropy'])
            for g in exp.get("groups", []) or []:
                if hasattr(g, "real_entropy") and g.real_entropy is not None:
                    buckets[target_ent].append(float(g.real_entropy))

        if not buckets:
            print("No matching experiments found for chosen weight vector.")
            return None, None

        target_vals = sorted(buckets.keys())
        data = [buckets[t] for t in target_vals]

        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(data, labels=[f"{t:.2f}" for t in target_vals])
        ax.set_xlabel("Target Entropy")
        ax.set_ylabel("Realized Group Entropy")
        if title:
            ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=200, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

