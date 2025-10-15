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
    # Helper: Count transitions
    # ───────────────────────────────────────────
    def _count_transitions(self, history):
        H, M, D = history[:, 0], history[:, 1], history[:, 2]
        dH, dM, dD = np.diff(H), np.diff(M), np.diff(D)
        transitions = {
            "H→M": int(np.sum(np.maximum(-dH, 0))),
            "H→D": int(np.sum(np.maximum(-dH - np.maximum(dM, 0), 0))),
            "M→D": int(np.sum(np.maximum(-dM, 0))),
            "M→H": int(np.sum(np.maximum(dH, 0))),
            "D→M": int(np.sum(np.maximum(dM, 0))),
            "D→H": int(np.sum(np.maximum(dH, 0))),
        }
        return transitions

    # ───────────────────────────────────────────
    # Helper: Compute weighted density
    # ───────────────────────────────────────────
    def _calculate_density(self, group):
        """
        Computes mean edge weight (normalized density) for the group's network.
        Since all graphs are fully connected, density ≈ average tie strength.
        """
        G = group.network
        weights = [d.get("weight", 0) for _, _, d in G.edges(data=True)]
        if len(weights) == 0:
            return 0
        return np.mean(weights)

    # ───────────────────────────────────────────
    # Extract Transition Data
    # ───────────────────────────────────────────
    def extract_transition_data(self, use_density=False):
        """
        Computes per-cohort average transitions and adds either
        'trait_entropy' or 'density' as x-axis variable.

        Parameters
        ----------
        use_density : bool
            If True, use network density on x-axis.
            If False, use target trait entropy.
        """
        if self.experiment_data is None:
            raise RuntimeError("No experiment data loaded. Run load_experiment_data() first.")

        records = []
        for _, exp in self.experiment_data.items():
            params = exp["params"]
            seed = params.get("seed")
            target_entropy = params.get("target_entropy")
            contagion_histories = exp.get("contagion_histories", {})
            groups = exp.get("groups", [])

            group_transitions = []
            densities = []

            # loop over groups
            for group in groups:
                densities.append(self._calculate_density(group))
                hist = contagion_histories.get(group.group_id)
                if hist is not None:
                    trans = self._count_transitions(hist)
                    group_transitions.append(trans)

            if group_transitions:
                avg_trans = {k: np.mean([t[k] for t in group_transitions]) for k in group_transitions[0]}
                avg_density = np.mean(densities)
                records.append({
                    "seed": seed,
                    "trait_entropy": target_entropy,
                    "density": avg_density,
                    **avg_trans
                })

        return records

    # ───────────────────────────────────────────
    # Plotting
    # ───────────────────────────────────────────
    def plot_transitions(self, records, x_axis="trait_entropy", band=False):
        """
        Plot mean ± SD of transitions vs. chosen x-axis.
        x_axis: "trait_entropy" or "density"
        band:   False -> error bars; True -> shaded ±1 SD band
        """
        import pandas as pd
        df = pd.DataFrame(records)
        if x_axis not in ["trait_entropy", "density"]:
            raise ValueError("x_axis must be 'trait_entropy' or 'density'")

        # aggregate across seeds (records are already averaged across groups per cohort)
        agg = df.groupby(x_axis).agg(
            {t: ['mean', 'std'] for t in ["H→M", "H→D", "M→D", "M→H", "D→M", "D→H"]}
        )
        agg.columns = [f"{t}_{stat}" for t, stat in agg.columns]
        agg = agg.reset_index()

        xs = agg[x_axis].values
        trans = ["H→M", "H→D", "M→D", "M→H", "D→M", "D→H"]

        plt.figure(figsize=(10, 6))
        for t in trans:
            m = agg[f"{t}_mean"].values
            s = agg[f"{t}_std"].values

            if band:
                plt.plot(xs, m, marker="o", label=t)
                plt.fill_between(xs, m - s, m + s, alpha=0.2)
            else:
                plt.errorbar(xs, m, yerr=s, fmt='-o', capsize=4, label=t)

        plt.xlabel("Network density" if x_axis == "density" else "Target trait entropy")
        plt.ylabel("Average number of transitions (±1 SD across seeds)")
        plt.title("Average Depression State Transitions (averaged across groups → seeds)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        # plt.ylim(20, 38)
        plt.tight_layout()
        plt.show()


    def plot_final_state_triangles(self, figsize=(15, 6), show_ticks=True, shared_scale=True):
        """
        Three side-by-side triangular (ternary-style) scatter plots (pure matplotlib).
        Each point = initial composition (H, M, D); color = final count of one state.
        Fixed so that grid lines and tick labels align with the correct components.

        Parameters
        ----------
        show_ticks : bool
            Draw ternary grid lines and integer tick labels (1..total-1) on all three axes.
        shared_scale : bool
            If True, all three colorbars share [0, total] to ease comparison.
        """
        if self.experiment_data is None:
            raise RuntimeError("No experiment data loaded. Run load_experiment_data() first.")

        # ── collect initial & final counts
        init_points, finals = [], []
        for exp in self.experiment_data.values():
            contagion_histories = exp.get("contagion_histories", {})
            groups = exp.get("groups", [])
            for g in groups:
                hist = contagion_histories.get(g.group_id)
                if hist is None or len(hist) == 0:
                    continue
                initial, final = hist[0], hist[-1]      # [H0, M0, D0], [Hf, Mf, Df]
                init_points.append(tuple(initial))
                finals.append(tuple(final))

        if not init_points:
            raise ValueError("No valid histories found in experiment data.")

        init_points = np.array(init_points)            # shape [N, 3]
        finals = np.array(finals)                      # shape [N, 3]
        total = int(np.sum(init_points[0]))            # group size (e.g., 10)

        # ── barycentric (H,M,D) → Cartesian (x,y) for an equilateral triangle
        def bary_to_xy(h, m, d):
            # bottom edge (d=0): x = m/total, y = 0
            x = (m + 0.5 * d) / total
            y = (np.sqrt(3) / 2.0) * (d / total)
            return x, y

        XY = np.array([bary_to_xy(h, m, d) for (h, m, d) in init_points])

        # ── figure scaffold
        titles = ["Healthy at End", "Mild at End", "Depressed at End"]
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # triangle boundary (H,M,D = (total,0,0), (0,total,0), (0,0,total))
        tri = np.array([
            bary_to_xy(total, 0, 0),   # all Healthy  (bottom-left)
            bary_to_xy(0, total, 0),   # all Mild     (bottom-right)
            bary_to_xy(0, 0, total),   # all Depressed(top)
            bary_to_xy(total, 0, 0)
        ])

        # optional shared color scale for easier side-by-side comparison
        vmin = 0 if shared_scale else None
        vmax = total if shared_scale else None

        for ax, col_idx, title in zip(axes, range(3), titles):
            vals = finals[:, col_idx]
            cmin = vmin if vmin is not None else vals.min()
            cmax = vmax if vmax is not None else vals.max()
            norm = (vals - cmin) / (cmax - cmin + 1e-9)

            # scatter of initial compositions, colored by final count of the chosen state
            sc = ax.scatter(XY[:, 0], XY[:, 1], c=norm, cmap="viridis",
                            s=80, edgecolor="k")

            # draw triangle boundary
            ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1.3)

            # ── properly aligned grid/ticks
            if show_ticks:
                tick_vals = np.arange(1, total)
                # constant Healthy lines: H = k (segment from (k, total-k, 0) to (k, 0, total-k))
                for k in tick_vals:
                    p1 = bary_to_xy(k, total - k, 0)
                    p2 = bary_to_xy(k, 0, total - k)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", lw=0.6, alpha=0.45)
                # constant Mild lines: M = k (segment from (total-k, k, 0) to (0, k, total-k))
                for k in tick_vals:
                    p1 = bary_to_xy(total - k, k, 0)
                    p2 = bary_to_xy(0, k, total - k)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", lw=0.6, alpha=0.45)
                # constant Depressed lines: D = k (segment from (total-k, 0, k) to (0, total-k, k))
                for k in tick_vals:
                    p1 = bary_to_xy(total - k, 0, k)
                    p2 = bary_to_xy(0, total - k, k)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", lw=0.6, alpha=0.45)

                # edge tick labels aligned with their components:
                # Mild ticks along the bottom edge (d=0): (h=total-k, m=k, d=0) → x = k/total, y=0
                for k in tick_vals:
                    xb, yb = bary_to_xy(total - k, k, 0)
                    ax.text(xb, yb - 0.06, f"{k}", ha="center", va="center",
                            fontsize=8, color="gray")

                # Depressed ticks along the right edge (h=0): (0, total-k, k)
                for k in tick_vals:
                    xr, yr = bary_to_xy(0, total - k, k)
                    ax.text(xr + 0.05, yr, f"{k}", ha="left", va="center",
                            fontsize=8, color="gray")

                # Healthy ticks along the left edge (m=0): (k, 0, total-k)
                for k in tick_vals:
                    xl, yl = bary_to_xy(k, 0, total - k)
                    ax.text(xl - 0.05, yl, f"{k}", ha="right", va="center",
                            fontsize=8, color="gray")

            # corner labels
            off = 0.055
            ax.text(tri[0, 0] - off, tri[0, 1] - off, "Healthy", ha="center", va="center", fontsize=10)
            ax.text(tri[1, 0] + off, tri[1, 1] - off, "Mild", ha="center", va="center", fontsize=10)
            ax.text(tri[2, 0],       tri[2, 1] + off, "Depressed", ha="center", va="center", fontsize=10)

            ax.set_title(title, fontsize=12)
            ax.set_aspect("equal")
            ax.axis("off")

            # colorbar with correct numeric range
            sm = plt.cm.ScalarMappable(cmap="viridis",
                                       norm=plt.Normalize(vmin=cmin, vmax=cmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"Final {title.split()[0]} count")

        plt.suptitle("Final State Composition by Initial Group Configuration", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_fraction_triangle_2_axes(self, figsize=(15, 5), share_scale=True):
        """
        Plots three right-triangle heatmaps (fractions) using only two axes:
        x = initial fraction mildly depressed
        y = initial fraction depressed
        (Healthy = 1 - x - y)

        Each triangle shows the final fraction of Healthy, Mild, or Depressed individuals.
        """
        if self.experiment_data is None:
            raise RuntimeError("No experiment data loaded. Run load_experiment_data() first.")

        # ─── collect initial and final data ────────────────────────────────
        data = []
        for exp in self.experiment_data.values():
            contagion_histories = exp.get("contagion_histories", {})
            groups = exp.get("groups", [])
            for g in groups:
                hist = contagion_histories.get(g.group_id)
                if hist is None or len(hist) == 0:
                    continue
                initial, final = hist[0], hist[-1]
                total = np.sum(initial)
                frac_init = initial / total
                frac_final = final / total
                data.append((*frac_init, *frac_final))  # (H0,M0,D0,Hf,Mf,Df)

        if not data:
            raise ValueError("No valid histories found in experiment data.")

        data = np.array(data)
        M0, D0 = data[:, 1], data[:, 2]   # initial fractions
        Hf, Mf, Df = data[:, 3], data[:, 4], data[:, 5]  # final fractions

        # ─── prepare grid ─────────────────────────────────────────────────
        # only valid region where M0 + D0 ≤ 1
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
        titles = ["Healthy at End", "Mild at End", "Depressed at End"]
        finals = [Hf, Mf, Df]

        vmin = 0 if share_scale else None
        vmax = 1 if share_scale else None

        for ax, title, Z in zip(axes, titles, finals):
            # build triangular mask
            sc = ax.scatter(M0, D0, c=Z, cmap='plasma', vmin=vmin, vmax=vmax,
                            s=80, edgecolor='k', alpha=0.9)
            # diagonal line where M+D=1 (no healthy left)
            ax.plot([0, 1], [1, 0], 'k--', lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

            # labels
            ax.set_xlabel("Initial fraction Mildly Depressed")
            ax.set_ylabel("Initial fraction Depressed")
            ax.set_title(title)

            # colorbar
            sm = plt.cm.ScalarMappable(cmap="plasma",
                                       norm=plt.Normalize(vmin=vmin if vmin is not None else Z.min(),
                                                          vmax=vmax if vmax is not None else Z.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Final fraction")

        plt.suptitle("Final Fraction of Each State vs Initial Group Composition", fontsize=14)
        plt.tight_layout()
        plt.show()




