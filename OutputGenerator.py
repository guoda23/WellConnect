import os
import re
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
from pathlib import Path
import gc, math
from tqdm import tqdm

from Visualizer3DScatterplot import Visualizer3DScatterPlot


class OutputGenerator:
    def __init__(self, batch_folder, mode):
        """
        Visualization and analysis tools for experiment results.
        Parameters
        ----------
        batch_folder : str
            Path to the batch folder containing experiment subfolders.
        mode : {"deterministic", "stochastic"}
            Type of experiments in the batch folder.
        """
        self.batch_folder = batch_folder
        self.experiment_data = None

        if mode not in ("deterministic", "stochastic"):
            raise ValueError("mode must be 'deterministic' or 'stochastic'")
        self.mode = mode


    def _load_experiment_data(self, seeds=None, noise_level=None):
        """
        Load experiment `.pkl` files under the batch folder by directly constructing paths.
        For stochastic case one noise level must be specified
        Heatmap generation method helper.

        Parameters
        ----------
        seeds : list[int] or None
            If given, only load experiments with these seeds, otherwise all that are available.
        noise_level : float or None
            (Only for stochastic) If given, only load experiments with that noise levels.

        Returns
        -------
        dict
            Mapping {filepath: experiment_data}.
        """
        base = Path(self.batch_folder)
        all_experiments_data = {}

        if self.mode == "deterministic":
            seed_dirs = [base / f"seed_{s}" for s in (seeds or [])] if seeds else list(base.glob("seed_*"))

            for sdir in tqdm(seed_dirs, desc="Loading seeds (deterministic)"):
                for run_dir in sdir.glob("experiment_run_*"):
                    for pkl_file in run_dir.glob("*.pkl"):
                        with open(pkl_file, "rb") as f:
                            experiment_data = pickle.load(f)
                        # deterministic → noise_std = None
                        experiment_data["params"].setdefault("noise_std", None)
                        all_experiments_data[str(pkl_file)] = experiment_data

        elif self.mode == "stochastic":
            if noise_level is None:
                    raise ValueError("For stochastic mode, you must specify one noise_level.")
            
            seed_dirs = [base / f"seed_{s}" for s in (seeds or [])] if seeds else list(base.glob("seed_*"))

            for sdir in tqdm(seed_dirs, desc=f"Loading seeds (stochastic, noise={noise_level})"):
                noise_dir = sdir / f"noise_{noise_level}"

                try:
                    noise_val = float(str(noise_dir).split("noise_")[1])
                except Exception:
                    noise_val = None

                for run_dir in noise_dir.glob("experiment_run_*"):
                    for pkl_file in run_dir.glob("*.pkl"):
                        with open(pkl_file, "rb") as f:
                            experiment_data = pickle.load(f)
                        experiment_data["params"].setdefault("noise_std", noise_val)
                        all_experiments_data[str(pkl_file)] = experiment_data

        self.experiment_data = all_experiments_data
        return all_experiments_data


    def extract_metrics(self, trait_of_interest, stat_power_measure='absolute_error',
                        target_entropy=True, seeds=None, noise_levels=None):
        """
        Extract per-group evaluation metrics from experiments, optionally filtering 
        by seed and/or noise level.

        For each experiment that passes the filters, this method collects the 
        error (or other statistical power measure) for the given trait across all groups,
        together with contextual information (seed, noise_std, entropies, weights).

        Parameters
        ----------
        trait_of_interest : str
            The trait name (key in measure_dict) for which to extract metrics.
        stat_power_measure : str, default 'absolute_error'
            Which metric to extract for the trait (e.g., 'absolute_error', 'bias').
        target_entropy : bool, default True
            If True, use the target entropy value from params.
            If False, use the realized entropy of each group.
        seeds : list[int] or None, default None
            If provided, only experiments with a matching seed are included.
        noise_levels : list[float] or None, default None
            If provided, only experiments with a matching noise_std are included.
            Deterministic experiments have noise_std=None.

        Returns
        -------
        list of dict
            Each dict corresponds to one group and includes:
                - 'seed', 'noise_std'
                - 'weight_entropy', 'trait_entropy'
                - 'stat_power' (error value for this group)
                - 'group', 'recovered_weights_df'
                - 'true_weights', 'row_of_interest_in_table'
        """

        data = []

        for folder, experiment in self.experiment_data.items():
            seed_val = experiment['params'].get('seed')
            # filter by seed if given
            if seeds is not None and seed_val not in seeds:
                continue
            
            # filter by noise level if given
            noise_std = experiment['params'].get('noise_std', None)  
            if noise_levels is not None and noise_std not in noise_levels:
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


    def plot_heatmaps(
        self, traits, stat_power_measure="absolute_error", dependent_variable="mean",
        x_label="Weight Entropy", y_label="Trait Entropy",
        target_entropy=True, cmap="viridis", vmin=None, vmax=None,
        suptitle=False, annotate=False, fmt=".3f",
        figsize_per_plot=(5, 4), share_scale=True,
        snap_decimals=6, tick_decimals=3,
        seeds=None, noise_levels=None,
        export=True, save_path=None
    ):

        plt.rcParams.update({
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 22
        })

        def snap(v, d=snap_decimals):
            return float(np.round(float(v), d))

        traits = list(traits)
        all_x, all_y = set(), set()
        per_trait_df = []

        for trait in traits:
            rows = []
            for d in self.extract_metrics(
                trait_of_interest=trait, stat_power_measure=stat_power_measure,
                target_entropy=target_entropy, seeds=seeds, noise_levels=noise_levels
            ):
                rows.append({
                    "x": snap(d["weight_entropy"]),
                    "y": snap(d["trait_entropy"]),
                    "z": float(d["stat_power"])
                })
            df = pd.DataFrame(rows)
            if df.empty:
                per_trait_df.append((trait, df))
                continue
            all_x.update(df["x"].unique().tolist())
            all_y.update(df["y"].unique().tolist())
            per_trait_df.append((trait, df))

        if not all_x or not all_y:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            plt.show()
            return fig, [ax]

        x_vals = sorted(all_x)
        y_vals = sorted(all_y)
        grids = []

        for trait, df in per_trait_df:
            if df.empty:
                grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
            else:
                if dependent_variable == "mean":
                    dfg = df.groupby(["y", "x"], as_index=False)["z"].mean()
                elif dependent_variable == "variance":
                    dfg = df.groupby(["y", "x"], as_index=False)["z"].var()
                elif dependent_variable == "std":
                    dfg = df.groupby(["y", "x"], as_index=False)["z"].std()
                else:
                    raise ValueError(f"Unknown dependent_variable: {dependent_variable}")

                pivot = dfg.pivot(index="y", columns="x", values="z")
                pivot = pivot.reindex(index=y_vals, columns=x_vals)
                grid = pivot.to_numpy(dtype=float)
            grids.append((trait, grid))

        if (vmin is None or vmax is None) and share_scale:
            stacked = np.concatenate([g[1].ravel() for g in grids])
            stacked = stacked[~np.isnan(stacked)]
            if stacked.size:
                if vmin is None: vmin = float(np.nanmin(stacked))
                if vmax is None: vmax = float(np.nanmax(stacked))

        measure_label = stat_power_measure.replace("_", " ").title().capitalize()
        if dependent_variable == "mean":
            stat_label = f"Mean {measure_label}"
        elif dependent_variable == "std":
            stat_label = f"Std. {measure_label}"
        elif dependent_variable == "variance":
            stat_label = f"Variance {measure_label}"
        else:
            stat_label = f"{dependent_variable.capitalize()} of {measure_label}"

        # --- dynamic vertical sizing ---
        per_tick_height = 0.18  # inch per y-tick
        fig_height = max(figsize_per_plot[1], len(y_vals) * per_tick_height)

        fig, axes = plt.subplots(
            1, len(traits),
            figsize=(figsize_per_plot[0]*len(traits) + 1.5, fig_height),
            squeeze=False
        )
        axes = axes[0]

        ims = []
        for i, (ax, (trait, grid)) in enumerate(zip(axes, grids)):
            im = ax.imshow(grid, origin="lower", aspect="auto",
                        cmap=cmap, vmin=vmin, vmax=vmax)
            ims.append(im)

            ax.set_xticks(np.arange(len(x_vals)))
            ax.set_yticks(np.arange(len(y_vals)))
            ax.set_xticklabels([f"{v:.{tick_decimals}f}" for v in x_vals],
                            rotation=45, ha="right")
            ax.set_yticklabels([f"{v:.{tick_decimals}f}" for v in y_vals],
                            rotation=0)

            # --- trim top/bottom whitespace ---
            ax.set_ylim(-0.5, len(y_vals) - 0.5)

            ax.set_xlabel(x_label)
            if i == 0:
                ax.set_ylabel(y_label)

            pretty = trait.split("_")[0]
            pretty = re.sub(r'(?<!^)(?=[A-Z])', ' ', pretty)
            ax.set_title(f"{stat_label}: {pretty}")

            if annotate:
                for r in range(len(y_vals)):
                    for c in range(len(x_vals)):
                        val = grid[r, c]
                        if not np.isnan(val):
                            ax.text(c, r, format(val, fmt),
                                    ha="center", va="center")

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(ims[0], cax=cbar_ax)

        if suptitle:
            fig.suptitle(f"{stat_label} of Regression Coefficients by Weight and Trait Entropy")

        plt.tight_layout(rect=[0, 0.05, 0.88, 0.95])
        if export==True and save_path is None:
            fig.savefig(f"Results/homophily_f_retrievability/heatmaps_{dependent_variable}.png", dpi=300, bbox_inches="tight")
        elif export==True and save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return fig, axes


    def plot_combined_heatmaps(
        self,
        img_mean="Results/homophily_f_retrievability/heatmaps_mean.png",
        img_std="Results/homophily_f_retrievability/heatmaps_std.png",
        combined_out="Results/homophily_f_retrievability/heatmaps_combined.png",
        title="Absolute Error of Regression Coefficients by Weight and Trait Entropy",
        figsize=(8, 8),
        show=True,
        ):
        """
        Combine two saved heatmap images (mean and std) into a single figure.

        Parameters
        ----------
        img_mean : str
            Path to the mean heatmap image.
        img_std : str
            Path to the std heatmap image.
        combined_out : str
            Path where the combined figure should be saved.
        title : str
            Suptitle for the figure.
        figsize : tuple
            Size of the combined figure.
        show : bool
            Whether to call plt.show() at the end.
        """
        import matplotlib.image as mpimg

        # Load the saved plot images
        img1 = mpimg.imread(img_mean)
        img2 = mpimg.imread(img_std)

        # Create a figure with 2 rows, 1 column
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Show first image
        ax1.imshow(img1)
        ax1.axis("off")

        # Show second image
        ax2.imshow(img2)
        ax2.axis("off")

        # Add a title for the whole figure
        fig.suptitle(title, fontsize=12, y=0.98)

        # Adjust margins and spacing
        fig.subplots_adjust(
            top=0.95,
            bottom=0.03,
            left=0.0,
            right=1.0,
            hspace=0.05,
            wspace=0.0,
        )

        fig.savefig(combined_out, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig, (ax1, ax2)


    # def plot_3d(self, trait_of_interest, x_label="Weight Entropy", y_label="Trait Entropy", z_label="Weight absolute error", target_entropy=True): #TODO: remove once interactive plot is ready (run_3d_visualization())
    #     """
    #     Create a 3D scatter plot using the extracted data.
    #     !Non-interactive!
    #     """
    #     data = self.extract_metrics(trait_of_interest=trait_of_interest, target_entropy=target_entropy)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")

    #     x_data = [d["weight_entropy"] for d in data]
    #     y_data = [d["trait_entropy"]  for d in data]
    #     z_data = [d["stat_power"] for d in data]

    #     print(len(x_data), len(y_data), len(z_data))
    #     # Scatter plot
    #     color_map = plt.cm.viridis 
    #     scatter = ax.scatter(x_data, y_data, z_data, c=z_data, cmap="viridis", marker="o")

    #     # Add axis labels and title
    #     ax.set_xlabel(x_label)
    #     ax.set_ylabel(y_label)
    #     ax.set_zlabel(z_label)
    #     plt.title("3D Visualization of Group Metrics")

    #     # Add a color bar to show the mapping of Z-axis values to colors
    #     cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, alpha=0.5)
    #     cbar.set_label(z_label)

    #     # Show the plot
    #     plt.show()


    def build_noise_error_summary(self, batch_folder,
                                cache_path=None,
                                stat_power_measure="absolute_error",
                                seeds=None,
                                flush_every=1000):
        """
        Efficiently aggregate mean/std of errors per noise level
        from experiment_*.pkl files, streaming to CSV with a progress bar.
        """

        base = Path(batch_folder)
        if cache_path is None:
            cache_path = base/"error_by_noise_summary_raw.csv"

        cache_path = Path(cache_path)
        if cache_path.exists():
            cache_path = cache_path.with_name(cache_path.stem + "(1)" + cache_path.suffix)


        acc = {}   # noise_std -> {"sum": x, "sum_sq": y, "count": n}
        rows = []

        # gather all pickle files up front so tqdm knows total
        all_pickles = list(base.rglob("experiment_*.pkl"))

        processed = 0
        for pkl_file in tqdm(all_pickles, desc="Aggregating experiments"):
            try:
                with open(pkl_file, "rb") as f:
                    exp = pickle.load(f)
            except Exception:
                continue

            # filter by seed if needed
            seed_val = exp["params"].get("seed")
            if seeds is not None and seed_val not in seeds:
                continue

            noise_std = exp["params"].get("noise_std")

            measure_dict = exp.get("measure_dict", {})
            for trait, metrics in measure_dict.items():
                vals = metrics.get(stat_power_measure, {})
                if isinstance(vals, dict):
                    for err in vals.values():
                        if noise_std not in acc:
                            acc[noise_std] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
                        acc[noise_std]["sum"] += err
                        acc[noise_std]["sum_sq"] += err**2
                        acc[noise_std]["count"] += 1

            del exp
            gc.collect()
            processed += 1

            # flush periodically to keep memory low
            if processed % flush_every == 0:
                rows.extend(self._acc_to_rows(acc))
                acc = {}
                pd.DataFrame(rows).to_csv(cache_path, mode="a",
                                        header=not Path(cache_path).exists(),
                                        index=False)
                rows = []

        # flush leftovers
        rows.extend(self._acc_to_rows(acc))
        if rows:
            pd.DataFrame(rows).to_csv(cache_path, mode="a",
                                    header=not Path(cache_path).exists(),
                                    index=False)

        print(f"✔ Saved summary to {cache_path} after {processed} experiments.")
        return pd.read_csv(cache_path)


    def clean_noise_summary(self, batch_folder, csv_in=None,
                            csv_out=None):
        # Aggregate the raw summary to one row per noise level.
        base = Path(batch_folder)
        if csv_in is None:
            csv_in = base/"error_by_noise_summary_raw.csv"

        if csv_out is None:
            csv_out = base/"error_by_noise_summary_clean.csv"

        # If file with same name exists add a (1) suffix
        csv_out = Path(csv_out)
        if csv_out.exists():
            csv_out = csv_out.with_name(csv_out.stem + "(1)" + csv_out.suffix)

        df = pd.read_csv(csv_in)

        # aggregate to one row per noise_std
        df_agg = df.groupby("noise_std").agg(
            mean=("mean", "mean"),      # mean of the per-batch means
            std=("mean", "std"),        # std of the per-batch means
            count=("count", "sum")      # total sample count
        ).reset_index()

        df_agg.to_csv(csv_out, index=False)
        print(f"✔ Cleaned summary written to {csv_out}")
        return df_agg


    def _acc_to_rows(self, acc):
        rows = []
        for noise, d in sorted(acc.items()):
            mean = d["sum"] / d["count"]
            var = d["sum_sq"] / d["count"] - mean**2
            std = math.sqrt(var) if var > 0 else 0.0
            rows.append({"noise_std": noise,
                        "mean": mean,
                        "std": std,
                        "count": d["count"]})
        return rows


    def plot_noise_vs_error(self, batch_folder, csv_path=None, save_path=None,
                            stat_power_measure="absolute_error", seeds=None,
                            export=True):
        """
        Plot Noise vs Avg Absolute Error using a pre-computed summary.
        """
        base = Path(batch_folder)

        if csv_path is None:
            csv_path = base/"error_by_noise_summary_clean.csv"

        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"Expected clean summary file at {csv_path}. "
                "Run build_noise_error_summary() and clean_noise_summary() first."
            )

        plt.rcParams.update({
            "axes.titlesize": 16,       
            "axes.labelsize": 14,        
            "xtick.labelsize": 12,     
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 22,
        })


        df = pd.read_csv(csv_path)

        fig, ax = plt.subplots(figsize=(7, 5))   # width=6 inches, height=4 inches

        ax.errorbar(
            df["noise_std"], df["mean"],
            yerr=df["std"],
            fmt="o-", capsize=5, markersize=7,
            color="dodgerblue",        # <── line + markers
            ecolor="deepskyblue",             # <── error bar color
            elinewidth=1.2,
            label="Mean ± 1 SD"

        )

        ax.set_xlabel("Noise Std (σ)", labelpad = 8)
        ax.set_ylabel("Average Absolute Error", labelpad = 8)
        ax.set_title("Effect of Gaussian Noise on Regression Error", pad = 12)
        ax.legend()

        if export:
            if save_path is None:
                save_path = "Results/homophily_f_retrievability/stochastic_error_by_noise.png"
            fig.savefig(
                save_path,
                dpi=300, bbox_inches="tight"
            )
            
        plt.tight_layout(pad=2.0)   # default is ~1.08
        plt.show()
        return fig, ax




    #TODO: make this work with the new experiment infrastructure
    # def run_3d_visualization(self, x_label="Weight Entropy", y_label="Trait Entropy", stat_power_measure='Absolute Error', trait_of_interest='Gender', target_entropy=True):
    #     """Runs the interactive 3D plot in the browser"""
    #     #capitalize strip
    #     z_label = f'{stat_power_measure.strip()} for {trait_of_interest}'
    #     data = self.extract_metrics(trait_of_interest=trait_of_interest, target_entropy=target_entropy)
    #     visualizer = Visualizer3DScatterPlot(data, x_label, y_label, z_label)
    #     visualizer.run()



    # def _weights_match(self, w_a: dict, w_b: dict, tol: float = 1e-9) -> bool:
    #     """Return True if two weight dicts match within tolerance (keys + values)."""
    #     if set(w_a.keys()) != set(w_b.keys()):
    #         return False
    #     for k in w_a.keys():
    #         if not math.isclose(float(w_a[k]), float(w_b[k]), rel_tol=0, abs_tol=tol):
    #             return False
    #     return True

    # def _make_uniform_weights_like(self, template_weights: dict) -> dict:
    #     """Build a uniform weight dict over the same keys as template_weights."""
    #     n = len(template_weights)
    #     if n == 0:
    #         return {}
    #     val = 1.0 / n
    #     return {k: val for k in template_weights.keys()}


    # def plot_real_entropy_boxplots(self, weight_choice=None, tol=1e-9, figsize=(7,4), title="Realized group entropies by target entropy (single weight choice)", savepath=None, show=True):
    #     """
    #     Make boxplots of realized group entropies (group.real_entropy) for each target_entropy,
    #     restricted to experiments that used a chosen weight vector.

    #     Parameters
    #     ----------
    #     weight_choice : dict or None
    #         Weight dict to filter experiments on (e.g., {'Age': 1/3, 'Gender': 1/3, 'Edu': 1/3}).
    #         If None, a uniform dict over the first experiment's weight keys is used.
    #     tol : float
    #         Tolerance for matching base_weights to `weight_choice`. Default 1e-9.
    #     figsize : tuple
    #         Matplotlib figure size (width, height). Default (7, 4).
    #     title : str or None
    #         Title for the plot. Use None for no title.
    #     savepath : str or None
    #         If given, saves the figure to this path (e.g. 'out/boxplots.png').
    #     show : bool
    #         If True, displays the plot with plt.show().
    #     """
    #     def weights_match(w_a, w_b):
    #         if set(w_a.keys()) != set(w_b.keys()):
    #             return False
    #         return all(abs(float(w_a[k]) - float(w_b[k])) <= tol for k in w_a)

    #     # default: uniform weights over first experiment keys
    #     if weight_choice is None:
    #         for exp in self.experiment_data.values():
    #             keys = list(exp['params']['base_weights'].keys())
    #             if keys:
    #                 val = 1.0 / len(keys)
    #                 weight_choice = {k: val for k in keys}
    #                 break

    #     buckets = defaultdict(list)

    #     for exp in self.experiment_data.values():
    #         base_w = exp['params']['base_weights']
    #         if not weights_match(base_w, weight_choice):
    #             continue

    #         target_ent = float(exp['params']['target_entropy'])
    #         for g in exp.get("groups", []) or []:
    #             if hasattr(g, "real_entropy") and g.real_entropy is not None:
    #                 buckets[target_ent].append(float(g.real_entropy))

    #     if not buckets:
    #         print("No matching experiments found for chosen weight vector.")
    #         return None, None

    #     target_vals = sorted(buckets.keys())
    #     data = [buckets[t] for t in target_vals]

    #     fig, ax = plt.subplots(figsize=figsize)
    #     ax.boxplot(data, labels=[f"{t:.2f}" for t in target_vals])
    #     ax.set_xlabel("Target Entropy")
    #     ax.set_ylabel("Realized Group Entropy")
    #     if title:
    #         ax.set_title(title)
    #     ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    #     plt.tight_layout()
    #     if savepath:
    #         fig.savefig(savepath, dpi=200, bbox_inches="tight")
    #     if show:
    #         plt.show()

    #     return fig, ax
    
#----------------------------------Input data (trait) analysis ----------------------------------
    def plot_trait_histograms(self, traits, csv_path="data/preprocessed.csv"):
        """
        Plots histograms for selected traits from the preprocessed dataset.

        Parameters
        ----------
        traits : list[str]
            List of trait/column names to plot.
        csv_path : str
            Path to the preprocessed dataset (default: 'data/preprocessed.csv').
        """

        # Increase global font sizes
        plt.rcParams.update({
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18
        })

        # Helper to clean names
        def prettify(name: str) -> str:
            # Split by underscore
            parts = name.split("_")

            # Remove "tertiary" if present
            parts = [p for p in parts if p.lower() != "tertiary"]

            # First part = base trait, split CamelCase
            base = re.sub(r'(?<!^)(?=[A-Z])', ' ', parts[0]).strip().title()

            # If extra parts exist, join them as category
            if len(parts) > 1:
                cat = " ".join([p.capitalize() for p in parts[1:]])
                return f"{base}: {cat}"
            else:
                return base

        # Load dataset
        df = pd.read_csv(csv_path)

        num_traits = len(traits)
        fig, axes = plt.subplots(1, num_traits, figsize=(6 * num_traits, 5), sharey=False)

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

                # Prettify x-tick labels for categorical traits
                new_labels = [prettify(lbl.get_text()) for lbl in ax.get_xticklabels()]
                ax.set_xticklabels(new_labels, rotation=45, ha="right")

            # Title at the top
            ax.set_title(prettify(trait))
            ax.set_xlabel("")   # no bottom xlabel
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()


    def plot_trait_collinearity(self, traits, csv_path="data/preprocessed.csv", save_path=None):
        """
        Plots a correlation heatmap and pairplot for selected traits.
        Works with both numeric and categorical traits by applying one-hot encoding.
        Labels are prettified for readability. Can also export figures.

        Parameters
        ----------
        traits : list[str]
            List of trait/column names to include.
        csv_path : str
            Path to the preprocessed dataset (default: 'data/preprocessed.csv').
        save_path : str or None
            If provided, path prefix to save figures (e.g. 'results/collinearity').
            Heatmap will be saved as '<save_path>_heatmap.png' and
            pairplot as '<save_path>_pairplot.png'.
        """
        import re

        # Load dataset
        df = pd.read_csv(csv_path)

        # Keep only selected traits
        df_sel = df[traits].copy()

        # One-hot encode categoricals
        df_num = pd.get_dummies(df_sel, drop_first=True)

        if df_num.empty:
            print("No usable traits found after encoding.")
            return

        # --- Prettify labels ---
        def prettify(col):
            # Split by underscore
            parts = col.split("_")

            # Remove "tertiary" if present
            parts = [p for p in parts if p.lower() != "tertiary"]

            # First part = base trait, add space before capital letters
            base = re.sub(r'([a-z])([A-Z])', r'\1 \2', parts[0]).strip().title()

            # If extra parts exist, join them as category
            if len(parts) > 1:
                cat = " ".join([p.capitalize() for p in parts[1:]])
                return f"{base}: {cat}"
            else:
                return base

        df_num = df_num.rename(columns={c: prettify(c) for c in df_num.columns})

        # 1. Correlation heatmap
        corr = df_num.corr()
        fig, ax = plt.subplots(figsize=(len(corr.columns) * 1.2, len(corr.columns) * 1.2))
        sns.heatmap(
            corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, square=True,
            ax=ax, cbar_kws={"shrink": 0.8}
        )

        # Rotate and enlarge labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

        # Title
        ax.set_title("Correlation Heatmap for Encoded Traits", fontsize=16, pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(f"{save_path}_heatmap.png", dpi=300, bbox_inches="tight")

        plt.show()

        # 2. Pairplot (only if not too many encoded variables)
        if df_num.shape[1] <= 5:
            g = sns.pairplot(df_num, diag_kind="hist")
            g.fig.suptitle("Pairwise Scatterplots (encoded traits)", y=1.02, fontsize=16)

            if save_path:
                g.savefig(f"{save_path}_pairplot.png", dpi=300, bbox_inches="tight")
            else:
                g.savefig(f"corr_plot.png", dpi=300, bbox_inches="tight")

            plt.show()
        else:
            print(f"Pairplot skipped: too many encoded variables ({df_num.shape[1]}).")




