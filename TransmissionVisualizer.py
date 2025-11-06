import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
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
    def _calculate_density(self, group, phq9_attr="PHQ9_Total", mode="all"):
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
        more: 
        Returns
        -------
        float
            Weighted network density between 0 and 1.
        """
        G = group.network
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0

        # Optional: compute only cross-state density
        if mode == "cross":
            total_weight = 0.0
            count = 0
            for a1, a2, d in G.edges(data=True):
                phq1 = getattr(a1, phq9_attr, None)
                phq2 = getattr(a2, phq9_attr, None)
                if phq1 is None or phq2 is None:
                    continue

                s1 = self.classify_phq9(phq1)
                s2 = self.classify_phq9(phq2)
                if s1 != s2:
                    total_weight += d.get("weight", 0.0)
                    count += 1

            if count == 0:
                return 0.0
            return total_weight / count


        #all edge weights
        total_weight = sum(d.get("weight", 0.0) for _, _, d in G.edges(data=True))
        possible_edges = n * (n - 1) / 2  # undirected
        return total_weight / possible_edges



    # ───────────────────────────────────────────
    # Plotting
    # ───────────────────────────────────────────

    def plot_relative_change_panels(self, mode="Mean", normalize=True, figsize=(15, 6),
                                    cmap="coolwarm", output_folder=None):
        """
        Plots three 2D heatmaps (Mild, Moderate, Severe) showing the relative change
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

                Mi0, Mo0, S0 = initial / total_i if normalize else initial
                MiF, MoF, SF = final / total_f if normalize else final

                dMi = (MiF - Mi0)
                dMo = (MoF - Mo0)
                dS = (SF - S0)

                records.append((Mo0, S0, dMi, dMo, dS))

                arr_norm = arr / np.sum(arr, axis=1, keepdims=True) if normalize else arr
                dep_var = np.var(arr_norm[:, 2])
                dep_variances.append((g.group_id, dep_var))
                all_histories.append((g.group_id, arr_norm))


        if not records:
            raise ValueError("No valid contagion histories found in experiment data.")

        # ─── Create DataFrame and aggregate ─────────────────────────────────
        df = pd.DataFrame(records, columns=["Mo0", "S0", "dMi", "dMo", "dS"])
        aggfunc_map = {"Mean": "mean", "Std": "std", "Counts": "count"}
        aggfunc = aggfunc_map.get(mode, "mean")

        pivots = {
            "Mild":      pd.pivot_table(df, values="dMi", index="Mo0", columns="S0", aggfunc=aggfunc),
            "Moderate":  pd.pivot_table(df, values="dMo", index="Mo0", columns="S0", aggfunc=aggfunc),
            "Severe":    pd.pivot_table(df, values="dS",  index="Mo0", columns="S0", aggfunc=aggfunc),
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

        # vmax, vmin = 0.008, -0.008
        vmax, vmin = 0.025, 0
        for ax, (title, pivot) in zip(axes, pivots.items()):
            im = ax.imshow(
                pivot.values,
                origin="lower",
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=[
                    pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(), pivot.index.max(),
                ],
            )

            ax.set_title(f"{title}", pad=15, fontsize=style["axes.titlesize"])
            ax.set_xlabel("Initial fraction Severely Depressed" if normalize else "Initial count Severely Depressed",
                        fontsize=style["axes.labelsize"])
            ax.set_ylabel("Initial fraction Moderately Depressed" if normalize else "Initial count Moderately Depressed",
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
            pivot = pivots["Severe"]

            # find cell with highest variance
            max_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
            max_M0 = pivot.index[max_idx[0]]
            max_D0 = pivot.columns[max_idx[1]]

            print(f"\nHighest variance cell in 'Severe':")
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

                    Mi0, Mo0, S0 = initial / total_i if normalize else initial
                    if np.isclose(Mo0, max_M0, atol=1e-3) and np.isclose(S0, max_D0, atol=1e-3):
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
                            cmap="viridis", output_folder=None, mode="all"):
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

                Mo = comp[1] / total
                S = comp[2] / total

                density = self._calculate_density(g, mode=mode)
                records.append((Mo, S, density))

        if not records:
            raise ValueError("No valid groups found for density heatmap.")

        # ─── Aggregate densities ─────────────────────────────────────────
        df = pd.DataFrame(records, columns=["Mo", "S", "Density"])
        df["Mo_rounded"] = df["Mo"].round(3)
        df["S_rounded"] = df["S"].round(3)

        grouped = df.groupby(["Mo_rounded", "S_rounded"])["Density"]
        df_mean = grouped.mean().reset_index(name="MeanDensity")
        df_std = grouped.std(ddof=0).reset_index(name="StdDensity")

        pivot_mean = pd.pivot_table(df_mean, values="MeanDensity",
                                    index="Mo_rounded", columns="S_rounded", aggfunc="mean")
        pivot_std = pd.pivot_table(df_std, values="StdDensity",
                                    index="Mo_rounded", columns="S_rounded", aggfunc="mean")

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
            vmax = 0.9

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
            ax.set_xlabel(f"{state.capitalize()} fraction Severely Depressed", fontsize=style["axes.labelsize"])
            ax.set_ylabel(f"{state.capitalize()} fraction Moderately Depressed", fontsize=style["axes.labelsize"])

            ax.tick_params(axis="x", labelsize=style["xtick.labelsize"])
            ax.tick_params(axis="y", labelsize=style["ytick.labelsize"])

            cbar = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
            cbar.ax.tick_params(labelsize=style["cbar.labelsize"],
                                length=style["cbar.ticklength"],
                                width=style["cbar.tickwidth"])

        plt.suptitle(f"Mean Network Density and Std by Group Composition ({mode} ties)",
                    fontsize=style["figure.titlesize"], y=1)
        plt.tight_layout()

        # ─── Optional export ────────────────────────────────────────────────
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            save_path = output_folder / f"density_heatmap_{state}_{mode}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        plt.show()

    # ────────────────────────────────────────────────
    # Helper: classify PHQ-9
    # ────────────────────────────────────────────────
    def classify_phq9(self, score):
        if score <= 9:
            return "Mild"
        elif score <= 14:
            return "Moderate"
        else:
            return "Severe"


    def plot_stacked_phq9_distributions(
        self,
        traits,
        csv_path="data/preprocessed.csv",
        phq9_col="PHQ9_Total",
        output_folder=None,
    ):
        """
        Plots stacked bar charts for multiple demographic traits showing
        raw counts of PHQ-9 severity categories (Mild, Moderate, Severe).
        """

        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path

        # ────────────────────────────────────────────────
        # Style settings (centralized)
        # ────────────────────────────────────────────────
        style = {
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "figure.titlesize": 25,
            "color_palette": ["#81C784", "#FFD54F", "#E57373"],  # Mild → Moderate → Severe
            "figure.figsize_unit": (8, 6.5),
        }

        # ────────────────────────────────────────────────
        # Category orders (confirmed)
        # ────────────────────────────────────────────────
        category_orders = {
            "Age_tertiary": ["Young", "Mid", "Older"],
            "EducationLevel_tertiary": ["Low", "Medium", "High"],
            "Gender_tertiary": ["Female", "Male", "Other"],
        }
        state_order = ["Mild", "Moderate", "Severe"]

        # ────────────────────────────────────────────────
        # Load and classify
        # ────────────────────────────────────────────────
        usecols = traits + [phq9_col]
        df = pd.read_csv(csv_path, usecols=usecols)
        df["PHQ9_Category"] = df[phq9_col].apply(self.classify_phq9)
        df = df[df["PHQ9_Category"] != "Unknown"]

        # Normalize casing per trait
        if "Age_tertiary" in df.columns:
            df["Age_tertiary"] = df["Age_tertiary"].str.title()
        if "EducationLevel_tertiary" in df.columns:
            df["EducationLevel_tertiary"] = df["EducationLevel_tertiary"].str.title()
        if "Gender_tertiary" in df.columns:
            df["Gender_tertiary"] = df["Gender_tertiary"].str.title()

        # ────────────────────────────────────────────────
        # Create figure and axes
        # ────────────────────────────────────────────────
        num_traits = len(traits)
        fig, axes = plt.subplots(
            1, num_traits, figsize=(style["figure.figsize_unit"][0] * num_traits, style["figure.figsize_unit"][1])
        )
        if num_traits == 1:
            axes = [axes]

        for ax, trait in zip(axes, traits):
            # Group and order
            counts = (
                df.groupby([trait, "PHQ9_Category"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=state_order, fill_value=0)
            )

            if trait in category_orders:
                order = category_orders[trait]
                counts = counts.loc[[x for x in order if x in counts.index]]

            # Plot
            counts.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=style["color_palette"],
                edgecolor="black",
                linewidth=1.0,
                legend=False,
            )

            # Style
            ax.set_title(
                trait.replace("_tertiary", "").replace("Level", ""),
                fontsize=style["axes.titlesize"],
                pad=10,
            )
            ax.set_xlabel("", fontsize=style["axes.labelsize"])
            ax.set_ylabel("Count", fontsize=style["axes.labelsize"], labelpad=15)
            ax.tick_params(axis="x", rotation=30, labelsize=style["xtick.labelsize"])
            ax.tick_params(axis="y", labelsize=style["ytick.labelsize"])

        # ────────────────────────────────────────────────
        # Shared legend and layout
        # ────────────────────────────────────────────────
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Depression\nSeverity",
            loc="center right",
            bbox_to_anchor=(0.99, 0.5),
            fontsize=style["legend.fontsize"],
            title_fontsize=style["legend.fontsize"],
        )

        fig.suptitle(
            "Trait Distributions by Depression Severity",
            fontsize=style["figure.titlesize"],
            y=0.97,
        )
        plt.tight_layout(rect=[0.02, 0, 0.9, 0.94])
        plt.subplots_adjust(wspace=0.25)

        # ────────────────────────────────────────────────
        # Save optional
        # ────────────────────────────────────────────────
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            save_path = output_folder / "trait_distributions_by_depression_state.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        plt.show()
