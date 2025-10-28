import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


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

    def plot_relative_change_panels(self, mode="raw", normalize=True, figsize=(15, 5), cmap="coolwarm"):
        """
        Plots three 2D heatmaps (Healthy, Mild, Depressed) showing the relative change
        between initial and final proportions of each state across all groups.

        X-axis: initial fraction (or count) of Depressed individuals
        Y-axis: initial fraction (or count) of Mildly Depressed individuals
        Color: relative change of each state

        Parameters
        ----------
        mode : {"raw", "std", "counts"}
            Determines what is shown in the heatmap:
                "raw"   → mean relative change (default)
                "std"   → standard deviation of relative change
                "counts"→ number of samples aggregated at that grid cell
        normalize : bool
            If True, use fractions (0–1) instead of raw counts.
        figsize : tuple
            Figure size for the three subplots.
        cmap : str
            Colormap for the heatmaps.
        """
        if self.experiment_data is None:
            raise RuntimeError("No experiment data loaded. Run load_experiment_data() first.")

        records = []
        eps = 1e-6

        # ─── Collect group-level data ──────────────────────────────────────
        for exp in self.experiment_data.values():
            contagion_histories = exp.get("contagion_histories", {})
            groups = exp.get("groups", [])

            for g in groups:
                runs = contagion_histories.get(g.group_id)
                if runs is None or len(runs) == 0:
                    continue

                # Handle nested dicts: {seed: array}
                if isinstance(runs, dict):
                    runs = list(runs.values())

                arr = np.array(runs)

                # Skip if empty or malformed
                if arr.size == 0:
                    continue

                # Handle multiple stochastic runs: (n_runs, timesteps, 3)
                if arr.ndim == 3:
                    mean_hist = np.mean(arr, axis=0)
                # Single run: (timesteps, 3)
                elif arr.ndim == 2:
                    mean_hist = arr
                # Single state vector (3,) → treat as no change
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

                dH = (Hf - H0) #/ (H0 + eps)
                dM = (Mf - M0) #/ (M0 + eps)
                dD = (Df - D0) #/ (D0 + eps)

                records.append((M0, D0, dH, dM, dD))

        if not records:
            raise ValueError("No valid contagion histories found in experiment data.")

        # ─── Create DataFrame and aggregate ─────────────────────────────────
        df = pd.DataFrame(records, columns=["M0", "D0", "dH", "dM", "dD"])

        aggfunc_map = {"raw": "mean", "std": "std", "counts": "count"}
        if mode not in aggfunc_map:
            raise ValueError("mode must be one of: 'raw', 'std', 'counts'")
        aggfunc = aggfunc_map[mode]

        pivots = {
            "Healthy":   pd.pivot_table(df, values="dH", index="M0", columns="D0", aggfunc=aggfunc),
            "Mild":      pd.pivot_table(df, values="dM", index="M0", columns="D0", aggfunc=aggfunc),
            "Depressed": pd.pivot_table(df, values="dD", index="M0", columns="D0", aggfunc=aggfunc),
        }

        # ─── Plot the three heatmaps ───────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
        vmin, vmax = -0.02, 0.005
        for ax, (title, pivot) in zip(axes, pivots.items()):
            im = ax.imshow(
                pivot.values,
                origin="lower",
                cmap=cmap,
                aspect="auto",
                vmin=vmin, vmax=vmax,
                extent=[
                    pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(), pivot.index.max()
                ]
            )

            ax.set_title(f"{title} ({mode})", pad=15)
            ax.set_xlabel("Initial fraction Depressed" if normalize else "Initial count Depressed")
            ax.set_ylabel("Initial fraction Mildly Depressed" if normalize else "Initial count Mildly Depressed")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar_label = {
                "raw": "Mean relative change",
                "std": "Std of relative change",
                "counts": "Sample count"
            }[mode]
            cbar.set_label(cbar_label)

        plt.suptitle(f"Relative Change by Initial Group Composition ({mode})", fontsize=14, y=1.03)
        plt.tight_layout()
        plt.show()


    def plot_density_heatmap(self, state="initial", normalize=True, figsize=(14, 6), cmap="viridis"):
        """
        Plots side-by-side 2D heatmaps:
            • Left  = mean network density
            • Right = standard deviation of density
        as a function of initial or final group composition.

        Handles nested contagion_histories structures and averages
        across multiple identical (M0, D0) configurations.

        Parameters
        ----------
        state : {"initial", "final"}
            Whether to plot against the initial or final group composition.
        normalize : bool
            If True, use fractions (0–1) instead of raw counts.
        figsize : tuple
            Figure size for the two subplots.
        cmap : str
            Colormap for the heatmaps.
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

                # Handle nested dicts: {seed: arr}
                if isinstance(hist, dict):
                    arrs = [np.array(v) for v in hist.values() if np.array(v).ndim >= 2]
                    if not arrs:
                        continue
                    # average across seeds → single (T,3)
                    hist = np.mean(arrs, axis=0)
                else:
                    hist = np.array(hist)

                if hist.ndim < 2 or hist.shape[0] < 1:
                    continue

                # pick composition
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

        # ─── Aggregate densities across identical (M,D) ───────────────
        df = pd.DataFrame(records, columns=["M", "D", "Density"])
        df["M_rounded"] = df["M"].round(3)
        df["D_rounded"] = df["D"].round(3)

        # Compute both mean and std
        grouped = df.groupby(["M_rounded", "D_rounded"])["Density"]
        df_mean = grouped.mean().reset_index(name="MeanDensity")
        df_std  = grouped.std(ddof=0).reset_index(name="StdDensity")

        pivot_mean = pd.pivot_table(
            df_mean, values="MeanDensity", index="M_rounded", columns="D_rounded", aggfunc="mean"
        )
        pivot_std = pd.pivot_table(
            df_std, values="StdDensity", index="M_rounded", columns="D_rounded", aggfunc="mean"
        )

        # ─── Plot side-by-side heatmaps ────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        titles = ["Mean Network Density", "Std. Dev. of Density"]
        pivots = [pivot_mean, pivot_std]

        for ax, pivot, title in zip(axes, pivots, titles):
            # Each subplot has its own color scale
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

            ax.plot([0, 1], [1, 0], "k--", lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{state.capitalize()} fraction Depressed")
            ax.set_ylabel(f"{state.capitalize()} fraction Mildly Depressed")
            ax.set_title(title, pad=15)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Network Density")

        plt.suptitle(f"{state.capitalize()} Network Density & Variability", fontsize=14)
        plt.tight_layout()
        plt.show()



