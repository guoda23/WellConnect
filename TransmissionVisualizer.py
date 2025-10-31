import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch import mode
from tqdm import tqdm
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import seaborn as sns
import re


class TransmissionVisualizer:
    def __init__(self, batch_folder):
        self.batch_folder = Path(batch_folder)
        self.experiment_data = None

    # ───────────────────────────────────────────
    # Load Experiment Data
    # ───────────────────────────────────────────
    def load_experiment_data(self, seeds=None, noise_level=None):
        base = Path(self.batch_folder)
        all_experiments = {}

        seed_dirs = [base / f"seed_{s}" for s in (seeds or [])] if seeds else list(base.glob("seed_*"))

        for sdir in tqdm(seed_dirs, desc="Loading seeds"):
            noise_dirs = [sdir / f"noise_{noise_level}"] if noise_level else list(sdir.glob("noise_*"))
            for ndir in noise_dirs:
                for run_dir in ndir.glob("experiment_run_*"):
                    for pkl_file in run_dir.glob("*.pkl"):
                        with open(pkl_file, "rb") as f:
                            exp_data = pickle.load(f)
                        all_experiments[str(pkl_file)] = exp_data

        self.experiment_data = all_experiments
        return all_experiments

    # ───────────────────────────────────────────
    # Helper: Compute weighted density
    # ───────────────────────────────────────────
    def _calculate_density(self, group):
        """
        Computes the weighted network density (mean tie strength) for an undirected,
        fully connected network with edge weights in [0, 1].

        Definition:
            D_w = (Σ_{i<j} w_ij) / (|V| * (|V|-1) / 2)

        This is the standard measure of weighted connectedness
        for undirected networks (Barrat et al., 2004).

        Parameters
        ----------
        group : Group
            Object containing a NetworkX undirected graph with 'weight' attributes on edges.

        Returns
        -------
        float
            Weighted network density between 0 and 1.
        """
        G = group.network
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0

        total_weight = sum(d.get("weight", 0.0) for _, _, d in G.edges(data=True))
        possible_edges = n * (n - 1) / 2  # undirected
        return total_weight / possible_edges



    # ───────────────────────────────────────────
    # Plotting
    # ───────────────────────────────────────────

    def plot_relative_change_panels(self, mode="Mean", normalize=True, figsize=(15, 6),
                                    cmap="coolwarm", output_folder=None):
        """
        Plots three 2D heatmaps (Healthy, Mild, Depressed) showing the relative change
        between initial and final proportions of each state across all groups.

        If 'output_folder' is provided, saves the figure as 'relative_change_{mode}.png'
        inside that folder.
        """

        if self.experiment_data is None:
            raise RuntimeError("No experiment data loaded. Run load_experiment_data() first.")

        records = []
        eps = 1e-6

        dep_variances = []
        all_histories = []

        # ─── Collect group-level data ──────────────────────────────────────
        for exp in self.experiment_data.values():
            contagion_histories = exp.get("contagion_histories", {})
            groups = exp.get("groups", [])

            for g in groups:
                runs = contagion_histories.get(g.group_id)
                if runs is None or len(runs) == 0:
                    continue

                if isinstance(runs, dict):
                    runs = list(runs.values())

                arr = np.array(runs)
                if arr.size == 0:
                    continue

                if arr.ndim == 3:
                    mean_hist = np.mean(arr, axis=0)
                elif arr.ndim == 2:
                    mean_hist = arr
                elif arr.ndim == 1 and arr.shape[0] == 3:
                    mean_hist = np.stack([arr, arr])
                else:
                    continue

                initial, final = mean_hist[0], mean_hist[-1]
                total_i, total_f = np.sum(initial), np.sum(final)
                if total_i == 0 or total_f == 0:
                    continue

                H0, M0, D0 = initial / total_i if normalize else initial
                Hf, Mf, Df = final / total_f if normalize else final

                dH = (Hf - H0)
                dM = (Mf - M0)
                dD = (Df - D0)

                records.append((M0, D0, dH, dM, dD))
                
                arr_norm = arr / np.sum(arr, axis=1, keepdims=True) if normalize else arr
                dep_var = np.var(arr_norm[:, 2])
                dep_variances.append((g.group_id, dep_var))
                all_histories.append((g.group_id, arr_norm))


        if not records:
            raise ValueError("No valid contagion histories found in experiment data.")

        # ─── Create DataFrame and aggregate ─────────────────────────────────
        df = pd.DataFrame(records, columns=["M0", "D0", "dH", "dM", "dD"])
        aggfunc_map = {"Mean": "mean", "Std": "std", "Counts": "count"}
        aggfunc = aggfunc_map.get(mode, "mean")

        pivots = {
            "Healthy":   pd.pivot_table(df, values="dH", index="M0", columns="D0", aggfunc=aggfunc),
            "Mild":      pd.pivot_table(df, values="dM", index="M0", columns="D0", aggfunc=aggfunc),
            "Depressed": pd.pivot_table(df, values="dD", index="M0", columns="D0", aggfunc=aggfunc),
        }

        # ─── Plot the three heatmaps ───────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

        # adjust all plotting sizes here
        style = {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "cbar.labelsize": 12,
            "cbar.ticklength": 6,
            "cbar.tickwidth": 1.2,
            "figure.titlesize": 20,
        }

        # vmax, vmin = 0.06, 0
        for ax, (title, pivot) in zip(axes, pivots.items()):
            im = ax.imshow(
                pivot.values,
                origin="lower",
                cmap=cmap,
                aspect="auto",
                # vmin=vmin,
                # vmax=vmax,
                extent=[
                    pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(), pivot.index.max(),
                ],
            )

            ax.set_title(f"{title}", pad=15, fontsize=style["axes.titlesize"])
            ax.set_xlabel("Initial fraction Depressed" if normalize else "Initial count Depressed",
                        fontsize=style["axes.labelsize"])
            ax.set_ylabel("Initial fraction Mildly Depressed" if normalize else "Initial count Mildly Depressed",
                        fontsize=style["axes.labelsize"])

            ax.tick_params(axis="x", labelsize=style["xtick.labelsize"])
            ax.tick_params(axis="y", labelsize=style["ytick.labelsize"])

            cbar = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
            cbar.ax.tick_params(labelsize=style["cbar.labelsize"],
                                length=style["cbar.ticklength"],
                                width=style["cbar.tickwidth"])

        plt.suptitle(f"{mode} Relative Change by Initial Group Composition",
                    fontsize=style["figure.titlesize"], y=1)
        plt.tight_layout()

        # ─── Optional export ────────────────────────────────────────────────
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            save_path = output_folder / f"relative_change_{mode}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        plt.show()


        # ─── Inspect groups contributing to highest variance cell ───────────────────────────
        if mode in {"Std", "Var"}:
            pivot = pivots["Depressed"]

            # find cell with highest variance
            max_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
            max_M0 = pivot.index[max_idx[0]]
            max_D0 = pivot.columns[max_idx[1]]

            print(f"\nHighest variance cell in 'Depressed':")
            print(f"   M0 = {max_M0:.3f}, D0 = {max_D0:.3f}")

            matching_groups = []
            for exp in self.experiment_data.values():
                contagion_histories = exp.get("contagion_histories", {})
                groups = exp.get("groups", [])

                for g in groups:
                    runs = contagion_histories.get(g.group_id)
                    if runs is None or len(runs) == 0:
                        continue

                    if isinstance(runs, dict):
                        runs = list(runs.values())

                    arr = np.array(runs)
                    if arr.ndim == 3:
                        mean_hist = np.mean(arr, axis=0)
                    elif arr.ndim == 2:
                        mean_hist = arr
                    else:
                        continue

                    initial = mean_hist[0]
                    total_i = np.sum(initial)
                    if total_i == 0:
                        continue

                    H0, M0, D0 = initial / total_i if normalize else initial

                    # match to cell
                    if np.isclose(M0, max_M0, atol=1e-3) and np.isclose(D0, max_D0, atol=1e-3):
                        matching_groups.append((g.group_id, mean_hist))

            print(f"\nGroups contributing to that cell: {len(matching_groups)} found.\n")
            for gid, hist in matching_groups:
                start = hist[0]
                end = hist[-1]
                print(f"Group {gid}: start = {np.round(start, 3)}, end = {np.round(end, 3)}")




    def combine_relative_change_plots(self, output_folder, spacing=80, bg_color=(255, 255, 255)):
        """
        Combines the Mean and Std relative change plots into a single vertically stacked image.

        Parameters
        ----------
        output_folder : str or Path
            Folder where the individual and combined plots are located.
        spacing : int, optional
            Vertical space (in pixels) between the Mean and Std images.
        bg_color : tuple, optional
            Background color for the combined image (RGB), default = white.
        """
        output_folder = Path(output_folder)
        mean_path = output_folder / "relative_change_Mean.png"
        std_path = output_folder / "relative_change_Std.png"
        combined_path = output_folder / "relative_change_combined.png"

        # ─── Check files exist ─────────────────────────────────────────────
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(
                f"Missing one or both required images:\n"
                f"  {mean_path if mean_path.exists() else '[missing]'}\n"
                f"  {std_path if std_path.exists() else '[missing]'}"
            )

        # ─── Open and combine ─────────────────────────────────────────────
        img_mean = Image.open(mean_path)
        img_std = Image.open(std_path)

        width = max(img_mean.width, img_std.width)
        height = img_mean.height + img_std.height + spacing

        combined = Image.new("RGB", (width, height), bg_color)
        combined.paste(img_mean, (0, 0))
        combined.paste(img_std, (0, img_mean.height + spacing))

        combined.save(combined_path)
        print(f"Combined figure saved to: {combined_path}")

        return combined_path


    def plot_density_heatmap(self, state="initial", normalize=True, figsize=(15, 6),
                            cmap="viridis", output_folder=None):
        """
        Plots side-by-side 2D heatmaps:
            • Left  = mean network density
            • Right = standard deviation of density
        as a function of initial or final group composition.

        Automatically adjusts axis limits to data (no forced 0–1 range).
        """

        if self.experiment_data is None:
            raise RuntimeError("No experiment data loaded. Run load_experiment_data() first.")
        if state not in {"initial", "final"}:
            raise ValueError("state must be 'initial' or 'final'")

        records = []

        # ─── Collect group data ─────────────────────────────────────────
        for exp in self.experiment_data.values():
            contagion_histories = exp.get("contagion_histories", {})
            groups = exp.get("groups", [])
            for g in groups:
                hist = contagion_histories.get(g.group_id)
                if hist is None:
                    continue

                if isinstance(hist, dict):
                    arrs = [np.array(v) for v in hist.values() if np.array(v).ndim >= 2]
                    if not arrs:
                        continue
                    hist = np.mean(arrs, axis=0)
                else:
                    hist = np.array(hist)

                if hist.ndim < 2 or hist.shape[0] < 1:
                    continue

                comp = hist[0] if state == "initial" else hist[-1]
                total = np.sum(comp)
                if total == 0:
                    continue

                M = comp[1] / total if normalize else comp[1]
                D = comp[2] / total if normalize else comp[2]

                density = self._calculate_density(g)
                records.append((M, D, density))

        if not records:
            raise ValueError("No valid groups found for density heatmap.")

        # ─── Aggregate densities ─────────────────────────────────────────
        df = pd.DataFrame(records, columns=["M", "D", "Density"])
        df["M_rounded"] = df["M"].round(3)
        df["D_rounded"] = df["D"].round(3)

        grouped = df.groupby(["M_rounded", "D_rounded"])["Density"]
        df_mean = grouped.mean().reset_index(name="MeanDensity")
        df_std = grouped.std(ddof=0).reset_index(name="StdDensity")

        pivot_mean = pd.pivot_table(df_mean, values="MeanDensity",
                                    index="M_rounded", columns="D_rounded", aggfunc="mean")
        pivot_std = pd.pivot_table(df_std, values="StdDensity",
                                index="M_rounded", columns="D_rounded", aggfunc="mean")

        # ─── Plot side-by-side heatmaps ────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        titles = ["Mean Network Density", "Std Network Density"]
        pivots = [pivot_mean, pivot_std]

        # STYLE SETTINGS — consistent with relative change panels
        style = {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "cbar.labelsize": 12,
            "cbar.ticklength": 6,
            "cbar.tickwidth": 1.2,
            "figure.titlesize": 20,
        }

        for ax, pivot, title in zip(axes, pivots, titles):
            vmin = pivot.min().min()
            vmax = pivot.max().max()

            im = ax.imshow(
                pivot.values,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                extent=[
                    pivot.columns.min(),
                    pivot.columns.max(),
                    pivot.index.min(),
                    pivot.index.max(),
                ],
            )

            # Automatically adjust axis ranges to data
            ax.set_xlim(pivot.columns.min(), pivot.columns.max())
            ax.set_ylim(pivot.index.min(), pivot.index.max())

            ax.set_title(title, pad=15, fontsize=style["axes.titlesize"])
            ax.set_xlabel(f"{state.capitalize()} fraction Depressed", fontsize=style["axes.labelsize"])
            ax.set_ylabel(f"{state.capitalize()} fraction Mildly Depressed", fontsize=style["axes.labelsize"])

            ax.tick_params(axis="x", labelsize=style["xtick.labelsize"])
            ax.tick_params(axis="y", labelsize=style["ytick.labelsize"])

            cbar = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
            cbar.ax.tick_params(labelsize=style["cbar.labelsize"],
                                length=style["cbar.ticklength"],
                                width=style["cbar.tickwidth"])

        plt.suptitle(f"Mean Network Density and Std by Group Composition",
                    fontsize=style["figure.titlesize"], y=1)
        plt.tight_layout()

        # ─── Optional export ────────────────────────────────────────────────
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            save_path = output_folder / f"density_heatmap_{state}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        plt.show()


    def plot_stacked_phq9_distributions(
        self,
        traits,
        phq9_col="PHQ9_Total",
        csv_path="data/preprocessed.csv",
        normalize=False, 
        output_folder=None
    ):
        """
        Plots stacked bar charts for multiple traits showing the distribution 
        of each trait across PHQ-9 depression categories (Mild / Moderate / Severe).

        Parameters
        ----------
        traits : list[str]
            List of trait/column names to plot (e.g. ['gender', 'age_tertiary', 'education_tertiary']).
        phq9_col : str
            Column name for PHQ-9 scores (default: 'PHQ9_Total').
        csv_path : str
            Path to the dataset.
        normalize : bool
            If True, show proportions (each bar sums to 1). If False, show raw counts.
        """

        # ────────────────────────────────────────────────
        # Helper: classify PHQ-9
        # ────────────────────────────────────────────────
        def classify_phq9(score):
            if pd.isna(score):
                return "Unknown"
            elif score <= 9:
                return "Mild"
            elif score <= 14:
                return "Moderate"
            else:
                return "Severe"

        # ────────────────────────────────────────────────
        # Helper: prettify names for x-axis labels
        # ────────────────────────────────────────────────
        def prettify(name: str) -> str:
            parts = name.split("_")
            parts = [p for p in parts if p.lower() != "tertiary"]
            base = re.sub(r"(?<!^)(?=[A-Z])", " ", parts[0]).strip().title()

            if base.lower().startswith("education"):
                base = "Education"
            if len(parts) > 1:
                cat = " ".join([p.capitalize() for p in parts[1:]])
                return f"{base}: {cat}"
            else:
                return base.capitalize()

        # ────────────────────────────────────────────────
        # Style settings
        # ────────────────────────────────────────────────
        style = {
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "figure.titlesize": 25,
            "color_palette": ["#81C784", "#FFD54F", "#E57373"],  # mild → moderate → severe
        }

        # ────────────────────────────────────────────────
        # Load and prepare data
        # ────────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        if phq9_col not in df.columns:
            raise ValueError(f"Column '{phq9_col}' not found in dataset.")

        df["PHQ9_Category"] = df[phq9_col].apply(classify_phq9)
        df = df[df["PHQ9_Category"] != "Unknown"]

        # Custom order of x-axis categories (ascending, exact labels)
        category_orders = {
            "age_tertiary": ["mid", "older", "young"],  # lowercase in your data
            "education_tertiary": ["Low", "Medium", "High"],
            "gender": ["Male", "Female", "Other"],
        }

        # ────────────────────────────────────────────────
        # Create subplots
        # ────────────────────────────────────────────────
        num_traits = len(traits)
        fig, axes = plt.subplots(1, num_traits, figsize=(8 * num_traits, 6.5))
        if num_traits == 1:
            axes = [axes]

        for ax, trait in zip(axes, traits):
            if trait not in df.columns:
                ax.set_visible(False)
                continue

            # Respect your preferred order
            if trait in category_orders:
                df[trait] = pd.Categorical(df[trait],
                                        categories=category_orders[trait],
                                        ordered=True)

            # Group and optionally normalize
            counts = df.groupby([trait, "PHQ9_Category"]).size().unstack(fill_value=0)
            data_to_plot = counts.div(counts.sum(axis=1), axis=0) if normalize else counts

            # Plot stacked bars
            data_to_plot.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=style["color_palette"],
                edgecolor="black",
                linewidth=1.0,
                legend=False,
            )

            # Prettify x labels
            new_labels = [lbl.get_text().capitalize() for lbl in ax.get_xticklabels()]
            ax.set_xticklabels(new_labels, rotation=30, ha="right", fontsize=style["xtick.labelsize"])

            # Titles and axis labels
            ax.set_title(prettify(trait), fontsize=style["axes.titlesize"], pad=10)
            ax.set_xlabel("", fontsize=style["axes.labelsize"])
            ax.set_ylabel("Proportion" if normalize else "Count", fontsize=style["axes.labelsize"])
            ax.tick_params(axis="y", labelsize=style["ytick.labelsize"])

        # ────────────────────────────────────────────────
        # Shared legend (right side, wrapped title)
        # ────────────────────────────────────────────────
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Depression\nsymptom\nseverity",  # wrapped neatly
            loc="center right",
            bbox_to_anchor=(0.99, 0.5),             # closer to center
            fontsize=style["legend.fontsize"],
            title_fontsize=style["legend.fontsize"],
        )

        # ────────────────────────────────────────────────
        # Title and layout adjustments
        # ────────────────────────────────────────────────
        fig.suptitle(
            "Trait Distributions by Depression Symptom Severity",
            fontsize=style["figure.titlesize"],
            y=0.98,  # lower title position (closer to plots)
        )

        # Adjust spacing to avoid any overlaps
        plt.tight_layout(rect=[0.02, 0, 0.9, 0.94])
        plt.subplots_adjust(wspace=0.25)

        # ─── Optional export ────────────────────────────────────────────────
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            save_path = output_folder / f"trait_distributions_by_depression_state.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        plt.show()
