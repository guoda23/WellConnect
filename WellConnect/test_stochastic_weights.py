# ───────────────────────────────
# Imports & Paths
# ───────────────────────────────
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve()
ROOT_DIR = SCRIPT_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "preprocessed.csv"

sys.path.append(str(ROOT_DIR))

from WellConnectController import WellConnectController

# ───────────────────────────────
# Test Parameters
# ───────────────────────────────
SEED = 20
TARGET_ENTROPY = 0.5
BASE_WEIGHTS = {"Age_tertiary": 1/3, "EducationLevel_tertiary": 1/3, "Gender_tertiary": 1/3}
ATTRIBUTES = list(BASE_WEIGHTS.keys())
MAX_DISTANCES = {k: 1 for k in ATTRIBUTES}
GROUP_SIZE = 10
NUM_GROUPS = 3
GROUP_FORMATION = "multi-trait-entropy"
NOISE_STDS = [0.2, 1.0]   # two values for testing

# ───────────────────────────────
# Main Test
# ───────────────────────────────
if __name__ == "__main__":
    controller = WellConnectController(
        data_path=DATA_PATH,
        group_size=GROUP_SIZE,
        attributes=ATTRIBUTES,
        max_distances=MAX_DISTANCES,
    )

    # Create groups once
    groups = controller.create_groups(
        strategy=GROUP_FORMATION,
        target_entropy=TARGET_ENTROPY,
        tolerance=float("inf"),
        traits=ATTRIBUTES,
        seed=SEED,
        num_groups=NUM_GROUPS,
    )

    # Ensure graphs exist
    for g in groups:
        g.create_group_graph()

    # Compare edge weights across noise levels
    for noise_std in NOISE_STDS:
        controller.run_on_groups(
            groups=groups,
            weights=BASE_WEIGHTS,
            homophily_function_name="linear_stochastic",
            drop_last_var=False,
            drop_var=None,
            regression_type="constrained",
            noise_std=noise_std,
        )

        # Get updated graph after weights are assigned
        G = groups[0].network
        edges = list(G.edges())[:5]  # take first 5 edges for clarity

        print(f"\nEdge weights with noise_std={noise_std}:")
        for u, v in edges:
            print(f"{u}-{v}: {G.edges[u, v]['weight']:.4f}")
