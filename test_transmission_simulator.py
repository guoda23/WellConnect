# ───────────────────────────────
# Quick HMDaModel test script
# ───────────────────────────────
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from TransmissionSimulator import TransmissionSimulator


def load_groups_from_pickle(pkl_path):
    """Load all groups from a saved experiment pickle."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["groups"]


def run_hmdh_on_group(group, steps=20, seed=20):
    """Run HMDaModel simulation on one group."""
    sim = TransmissionSimulator(model_type="HMDaModel", seed=seed)
    history, agents = sim.run(group, steps=steps)
    return np.array(history)


def plot_group_dynamics(histories, title="HMDaModel Simulation"):
    """Plot multiple group histories and their average trajectory."""
    labels = ["Healthy", "Mild", "Depressed"]
    colors = ["green", "orange", "red"]
    steps = np.arange(histories[0].shape[0])

    plt.figure(figsize=(8, 5))

    # plot each group faintly
    for hist in histories:
        for i, color in enumerate(colors):
            plt.plot(steps, hist[:, i], color=color, alpha=0.2)

    # mean trajectory
    mean_hist = np.mean(histories, axis=0)
    for i, (label, color) in enumerate(zip(labels, colors)):
        plt.plot(steps, mean_hist[:, i], color=color, lw=2.5, label=label)

    plt.xlabel("Time step")
    plt.ylabel("Number of individuals")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Path to one of your saved experiment .pkl files
    PKL_PATH = "Experiments/transmission/batch_2025-10-13_11-33-42/seed_1/noise_0/experiment_run_1_target_e_0.0_weight_e_1.5850/experiment_2025-10-13_11-33-44.pkl"

    # Load and simulate
    groups = load_groups_from_pickle(PKL_PATH)
    all_histories = [run_hmdh_on_group(g, steps=20) for g in groups]

    # Plot
    plot_group_dynamics(all_histories, title="HMDaModel dynamics (seed 1, entropy 1.2955)")
