import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy
import numpy as np
import networkx as nx
import math
from collections import defaultdict
import re
from pathlib import Path

from Visualizer3DScatterplot import Visualizer3DScatterPlot


class OutputGenerator:
    def __init__(self, batch_folder):
        self.batch_folder = batch_folder
        self.experiment_data = self._load_experiment_data()


    def _load_experiment_data(self):
        all_experiments_data = {}

        for root, _, files in os.walk(self.batch_folder):
            for file in files:
                if file.endswith(".pkl"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "rb") as f:
                        experiment_data = pickle.load(f)

                    # detect noise_std from folder name if available
                    noise_val = None
                    parts = Path(filepath).parts
                    for p in parts:
                        if p.startswith("noise_"):
                            try:
                                noise_val = float(p.split("_")[1])
                            except Exception:
                                pass

                    # enrich params with noise_std if not already present
                    if "params" in experiment_data and "noise_std" not in experiment_data["params"]:
                        experiment_data["params"]["noise_std"] = noise_val

                    all_experiments_data[filepath] = experiment_data

        return all_experiments_data


    

    def extract_metrics(self, trait_of_interest, stat_power_measure='absolute_error',
                        target_entropy=True, seeds=None):
        data = []

        for folder, experiment in self.experiment_data.items():
            seed_val = experiment['params'].get('seed')
            if seeds is not None and seed_val not in seeds:
                continue

            trait_entropy = experiment['params']['target_entropy']
            weight_dict = experiment['params']['base_weights']
            weight_entropy = self._calculate_entropy(weight_dict)
            noise_std = experiment['params'].get('noise_std', None)  # unified handling

            measure_dict = experiment["measure_dict"][trait_of_interest][stat_power_measure]
            groups_list = experiment["groups"]
            recovered_weights_df = experiment["recovered_weights_df"]

            for group in groups_list:
                group_id_within_cohort = group.group_id - 1
                group_absolute_error = measure_dict[group_id_within_cohort]
                trait_entropy_value = trait_entropy if target_entropy else group.real_entropy

                data.append({
                    "seed": seed_val,
                    "noise_std": noise_std,  # may be None
                    "weight_entropy": weight_entropy,
                    "trait_entropy": trait_entropy_value,
                    "stat_power": group_absolute_error,
                    "group": group,
                    "recovered_weights_df": recovered_weights_df,
                    "true_weights": weight_dict,
                    "row_of_interest_in_table": group_id_within_cohort
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
        target_entropy=True,  cmap="viridis", vmin=None, vmax=None,
        annotate=False, fmt=".3f", figsize_per_plot=(5, 4),
        share_scale=True, snap_decimals=6, tick_decimals=3
    ):
        # Font sizes
        plt.rcParams.update({
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 18
        })

        def snap(v, d=snap_decimals):
            # snap to a python float (not numpy scalar)
            return float(np.round(float(v), d))

        traits = list(traits)

        # First pass: collect snapped X/Y across ALL traits so subplots share axes
        all_x, all_y = set(), set()
        per_trait_df = []  # store (trait, df) pairs so we don't recompute

        for trait in traits:
            rows = []
            for d in self.extract_metrics(trait_of_interest=trait, target_entropy=target_entropy):
                rows.append({
                    "x": snap(d["weight_entropy"]),
                    "y": snap(d["trait_entropy"]),
                    "z": float(d["stat_power"])
                })
            df = pd.DataFrame(rows)
            if df.empty:
                # keep an empty df; it's handled below
                per_trait_df.append((trait, df))
                continue

            # Update global axes sets
            all_x.update(df["x"].unique().tolist())
            all_y.update(df["y"].unique().tolist())
            per_trait_df.append((trait, df))

        # If nothing to plot, bail gracefully
        if not all_x or not all_y:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            plt.show()
            return fig, [ax]

        # Shared sorted axis values (snapped)
        x_vals = sorted(all_x)
        y_vals = sorted(all_y)

        # Build one grid per trait (pivot + reindex to shared axes)
        grids = []
        for trait, df in per_trait_df:
            if df.empty:
                grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
            else:
                # average duplicates of the same (x,y)
                dfg = df.groupby(["y", "x"], as_index=False)["z"].mean()
                pivot = dfg.pivot(index="y", columns="x", values="z")

                # ensure full rectangular grid across shared axes
                pivot = pivot.reindex(index=y_vals, columns=x_vals)
                grid = pivot.to_numpy(dtype=float)
            grids.append((trait, grid))

        # Shared color scale if desired
        if (vmin is None or vmax is None) and share_scale:
            stacked = np.concatenate([g[1].ravel() for g in grids])
            stacked = stacked[~np.isnan(stacked)]
            if stacked.size:
                if vmin is None: vmin = float(np.nanmin(stacked))
                if vmax is None: vmax = float(np.nanmax(stacked))

        # Plotting
        n = len(traits)
        fig, axes = plt.subplots(
            1, n, figsize=(figsize_per_plot[0]*n + 1.5, figsize_per_plot[1]),
            squeeze=False
        )
        axes = axes[0]

        ims = []
        for i, (ax, (trait, grid)) in enumerate(zip(axes, grids)):
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ims.append(im)

            ax.set_xticks(np.arange(len(x_vals)))
            ax.set_yticks(np.arange(len(y_vals)))
            ax.set_xticklabels([f"{v:.{tick_decimals}f}" for v in x_vals], rotation=45, ha="right")
            ax.set_yticklabels([f"{v:.{tick_decimals}f}" for v in y_vals])

            ax.set_xlabel(x_label)
            if i == 0:
                ax.set_ylabel(y_label)

            # Clean trait title: cut at '_' and split CamelCase ("EducationLevel" -> "Education Level")
            pretty = trait.split("_")[0]
            pretty = re.sub(r'(?<!^)(?=[A-Z])', ' ', pretty)
            ax.set_title(f"Absolute error: {pretty}")

            if annotate:
                for r in range(len(y_vals)):
                    for c in range(len(x_vals)):
                        val = grid[r, c]
                        if not np.isnan(val):
                            ax.text(c, r, format(val, fmt), ha="center", va="center")

        # Shared colorbar on the right
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(ims[0], cax=cbar_ax)
        # no label: cbar.set_label("")

        fig.suptitle("Absolute Error of Regression Coefficients Across Weight and Trait Entropy")
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show()
        return fig, axes
    

    def plot_noise_vs_error(self, stat_power_measure='absolute_error',
                            target_entropy=True, seeds=None):
        """
        Plot average absolute error (aggregated across all traits, groups, entropies, weights, seeds)
        as a function of noise_std.
        """
        data = []

        for folder, experiment in self.experiment_data.items():
            seed_val = experiment['params'].get('seed')
            if seeds is not None and seed_val not in seeds:
                continue

            weight_dict = experiment['params']['base_weights']
            noise_std = experiment['params'].get('noise_std', None)

            # skip deterministic (no noise_std)
            if noise_std is None:
                continue

            groups_list = experiment["groups"]
            recovered_weights_df = experiment["recovered_weights_df"]

            # measure_dict has structure: trait -> stat_power_measure -> list[group_values]
            measure_dict = experiment["measure_dict"]

            # collect all errors across traits & groups
            errors = []
            for trait, metrics in measure_dict.items():
                values = metrics[stat_power_measure]
                errors.extend(values)

            if errors:
                avg_abs_error = float(np.mean(errors))
                data.append({
                    "seed": seed_val,
                    "noise_std": noise_std,
                    "avg_abs_error": avg_abs_error,
                    "true_weights": weight_dict,
                    "recovered_weights_df": recovered_weights_df
                })

        df = pd.DataFrame(data)
        grouped = df.groupby("noise_std")["avg_abs_error"].mean().reset_index()

        plt.scatter(grouped["noise_std"], grouped["avg_abs_error"])
        plt.xlabel("Noise Std")
        plt.ylabel("Average Absolute Error (all traits)")
        plt.title("Noise vs Absolute Error")
        plt.show()

        return grouped




    def run_3d_visualization(self, x_label="Weight Entropy", y_label="Trait Entropy", stat_power_measure='Absolute Error', trait_of_interest='Gender', target_entropy=True):
        """Runs the interactive 3D plot in the browser"""
        #TODO: check if this works with the new experiment infrastructure, make it work!
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
    

    def plot_trait_histograms(self, traits, csv_path="data/preprocessed.csv"):
        """
        Plots histograms for selected traits from the preprocessed dataset.

        Parameters:
            traits (list[str]): List of trait/column names to plot.
            csv_path (str): Path to the preprocessed dataset (default: 'data/preprocessed.csv').
        """
        # Load dataset
        df = pd.read_csv(csv_path)

        num_traits = len(traits)
        fig, axes = plt.subplots(1, num_traits, figsize=(5 * num_traits, 4), sharey=False)

        if num_traits == 1:
            axes = [axes]  # make iterable if only one plot

        for ax, trait in zip(axes, traits):
            if trait not in df.columns:
                ax.set_visible(False)
                continue

            if pd.api.types.is_numeric_dtype(df[trait]):
                sns.histplot(df[trait], kde=False, bins=10, ax=ax, color="#7E57C2")
            else:
                df[trait].value_counts().plot(kind="bar", ax=ax, color="#81C784")

            ax.set_title(f"Distribution of {trait}")
            ax.set_xlabel(trait)
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

