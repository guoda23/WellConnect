import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class SupercohortTransmissionVisualizer:
    """
    X-axis (default): average Shannon entropy of H/M/D distribution across groups.
        depression_state: 0→H, 1→M, 2→D
        entropy(group) = -Σ p_s log(p_s), normalized by log(3) so values ∈ [0,1]
        X = mean entropy across groups.

    Y-axis: mean ± std of transitions (summed over groups, aggregated over runs):
        H→M, H→D, M→D, and TOTAL = H→M + H→D + M→D.
    """

    TRANS_KEYS = ["H→M", "H→D", "M→D", "M→H", "D→M", "D→H"]
    WORSEN_KEYS = ["H→M", "H→D", "M→D"]

    def __init__(self, batch_folder: str):
        self.batch_folder = Path(batch_folder)
        self._per_seed_runs = {}

    # ───────────────────────────────────────────
    def load_batch(self, seeds=None, noise_level=None):
        root = self.batch_folder
        seed_dirs = [root / f"seed_{s}" for s in (seeds or [])] if seeds else sorted(root.glob("seed_*"))
        per_seed = {}

        for sdir in tqdm(seed_dirs, desc="Scanning seeds"):
            if not sdir.is_dir():
                continue
            seed_id = sdir.name
            noise_dirs = [sdir / f"noise_{noise_level}"] if noise_level else sorted(sdir.glob("noise_*"))
            exps = []
            for ndir in noise_dirs:
                if not ndir.is_dir():
                    continue
                for run_dir in sorted(ndir.glob("experiment_run_*")):
                    for pkl in run_dir.glob("*.pkl"):
                        with open(pkl, "rb") as f:
                            exp = pickle.load(f)
                        exps.append(exp)
            if exps:
                per_seed[seed_id] = exps

        self._per_seed_runs = per_seed
        return per_seed

    # ───────────────────────────────────────────
    @staticmethod
    def entropy_metric(groups):
        entropies = []
        for g in groups:
            G = g.network
            counts = {"H": 0, "M": 0, "D": 0}

            for agent in G.nodes():
                ds = getattr(agent, "depression_state", None)
                if ds == 0:
                    counts["H"] += 1
                elif ds == 1:
                    counts["M"] += 1
                elif ds == 2:
                    counts["D"] += 1

            total = sum(counts.values())
            if total == 0:
                continue

            probs = [v / total for v in counts.values() if v > 0]
            H = -sum(p * np.log(p) for p in probs) / np.log(3.0)
            entropies.append(H)

        return float(np.mean(entropies)) if entropies else np.nan

    # ───────────────────────────────────────────
    @classmethod
    def _sum_steps(cls, steps):
        totals = {}
        for sd in steps:
            for k, v in sd.items():
                totals[k] = totals.get(k, 0) + int(v)
        for k in cls.TRANS_KEYS:
            totals.setdefault(k, 0)
        return totals

    @classmethod
    def _runs_from_logs(cls, tlogs):
        if not tlogs:
            return []
        first_val = next(iter(tlogs.values()))
        out = []
        if isinstance(first_val, dict):
            for _, by_group in tlogs.items():
                run_totals = {}
                for steps in by_group.values():
                    stot = cls._sum_steps(steps)
                    for k, v in stot.items():
                        run_totals[k] = run_totals.get(k, 0) + v
                out.append(run_totals)
        else:
            run_totals = {}
            for steps in tlogs.values():
                stot = cls._sum_steps(steps)
                for k, v in stot.items():
                    run_totals[k] = run_totals.get(k, 0) + v
            out.append(run_totals)
        return out

    @classmethod
    def _mean_std(cls, run_totals_list):
        if not run_totals_list:
            zeros = {k: 0.0 for k in cls.TRANS_KEYS}
            return zeros.copy(), zeros.copy()

        arr = np.array([[run.get(k, 0) for k in cls.TRANS_KEYS] for run in run_totals_list], dtype=float)
        means = {k: float(np.mean(arr[:, i])) for i, k in enumerate(cls.TRANS_KEYS)}
        stds = {k: float(np.std(arr[:, i], ddof=1)) if arr.shape[0] > 1 else 0.0
                for i, k in enumerate(cls.TRANS_KEYS)}
        means["TOTAL"] = sum(means[k] for k in cls.WORSEN_KEYS)
        stds["TOTAL"]  = sum(stds[k]  for k in cls.WORSEN_KEYS)
        return means, stds

    # ───────────────────────────────────────────
    def extract_records(self, metric_fn=None):
        metric_fn = metric_fn or self.entropy_metric
        records = []
        for seed_id, exps in self._per_seed_runs.items():
            for exp in exps:
                groups = exp.get("groups", [])
                x_metric = metric_fn(groups)
                runs = self._runs_from_logs(exp.get("transition_logs", {}))
                means, stds = self._mean_std(runs)
                records.append({
                    "seed": exp.get("params", {}).get("seed", seed_id),
                    "metric": x_metric,
                    "H→M_mean": means["H→M"], "H→M_std": stds["H→M"],
                    "H→D_mean": means["H→D"], "H→D_std": stds["H→D"],
                    "M→D_mean": means["M→D"], "M→D_std": stds["M→D"],
                    "TOTAL_mean": means["TOTAL"], "TOTAL_std": stds["TOTAL"],
                })
        return records

    # ───────────────────────────────────────────
    def plot(self, records, x_label="Entropy (H/M/D)", title=None):
        df = pd.DataFrame(records)
        if df.empty or "metric" not in df.columns:
            print("[WARNING] No valid records to plot.")
            return

        df = df.replace([np.inf, -np.inf], np.nan)
        if df["metric"].isna().all():
            print("[WARNING] All metric values are NaN → X-axis cannot be plotted.")
            return

        df = df.dropna(subset=["metric"])
        x = df["metric"].values

        fig, ax = plt.subplots(figsize=(10, 6))
        for label in ["H→M", "H→D", "M→D", "TOTAL"]:
            ax.errorbar(
                x,
                df[f"{label}_mean"],
                yerr=df[f"{label}_std"],
                fmt='o',
                capsize=3,
                linestyle="none",
                label=label
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Transitions (mean ± SD)")
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

    def assortativity_weighted(self, groups):
        import numpy as np

        same_vals = []
        w_vals = []

        for g in groups:
            G = g.network
            for u, v, data in G.edges(data=True):
                w = data.get("weight", 0.0)
                su = getattr(u, "depression_state", None)
                sv = getattr(v, "depression_state", None)
                if su is None or sv is None:
                    continue
                same = 1.0 if su == sv else 0.0
                w_vals.append(w)
                same_vals.append(same)

        if len(w_vals) < 2:
            return np.nan

        w_arr = np.array(w_vals, dtype=float)
        same_arr = np.array(same_vals, dtype=float)

        if np.std(w_arr) == 0 or np.std(same_arr) == 0:
            return 0.0

        return float(np.corrcoef(w_arr, same_arr)[0, 1])


    def weighted_density(self, groups):
        densities = []
        for g in groups:
            G = g.network
            n = G.number_of_nodes()
            if n <= 1:
                continue
            total_weight = sum(data.get("weight", 0.0) for _, _, data in G.edges(data=True))
            possible_edges = n * (n - 1) / 2
            densities.append(total_weight / possible_edges)
        return float(np.mean(densities)) if densities else np.nan

