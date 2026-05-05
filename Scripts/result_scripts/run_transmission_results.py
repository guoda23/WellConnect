import sys
from pathlib import Path
from networkx import density
import pandas as pd


def resolve_project_root(start_dir: Path) -> Path:
    """Walk up from this script until we find the repository root."""
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "src" / "WellConnectController.py").exists() and (candidate / "Experiments").exists():
            return candidate
    raise RuntimeError("Could not locate project root from script location")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = resolve_project_root(SCRIPT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from src.modules.TransmissionVisualizer import TransmissionVisualizer

HOMOPHILY_FUNCTION = "amplified_effect" # 0.7 age, 0.2 edu, 0.1 gender      OR "even": 1/3 age, 1/3 edu, 1/3 gender

if HOMOPHILY_FUNCTION == "amplified_effect":
    BATCH_FOLDER = str(ROOT_DIR / "Experiments" / "transmission" / "batch_2025-10-31_17-50-13")
    OUTPUT_FOLDER = str(ROOT_DIR / "Results" / "transmission")
elif HOMOPHILY_FUNCTION == "even":
    BATCH_FOLDER = str(ROOT_DIR / "Experiments" / "transmission" / "batch_2025-10-31_19-10-40")
    OUTPUT_FOLDER = str(ROOT_DIR / "Results" / "transmission" / "even")



# Load experiments
viz = TransmissionVisualizer(batch_folder = BATCH_FOLDER)
viz.load_experiment_data(noise_level=0.15)

#General distribution of demographic traits x depression severity
viz.plot_stacked_phq9_distributions(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'],
                                    output_folder=OUTPUT_FOLDER)

# Count plot
viz.plot_group_count_distribution(output_folder=OUTPUT_FOLDER)

# General transmission dynamics plots
viz.plot_relative_change_panels(mode="Mean", vmax=0.008, vmin=-0.008, output_folder=OUTPUT_FOLDER)
viz.plot_relative_change_panels(mode="Std", vmax=0.025, vmin=0, output_folder=OUTPUT_FOLDER)

viz.combine_relative_change_plots(OUTPUT_FOLDER)

# Density heatmaps 2x2 mean and std for all and cross ties

viz.plot_density_heatmap(
    state="initial",
    output_folder=OUTPUT_FOLDER, #TODO: add /even for even weights
    mode="All"
)

viz.plot_density_heatmap(
    state="initial",
    output_folder=OUTPUT_FOLDER, #TODO: add /even for even weights
    mode="Cross"
)

viz.combine_density_heatmaps(OUTPUT_FOLDER)

# Boolean plot for top percentiles of density
grouped, conds = viz.plot_density_percentiles_2x2(state="initial", output_folder=OUTPUT_FOLDER)

# Plots: Boolean mask on contagion dynamics for the four stratgies (max all ties, max cross ties; min all ties, max cross ties; etc)
for label, condition_mask in conds.items():
    # build boolean pivot mask
    mask_pivot = pd.pivot_table(
        grouped.assign(ConditionMet=condition_mask),
        values="ConditionMet",
        index="Mo_rounded",
        columns="S_rounded",
        aggfunc=lambda x: any(x),
    ).astype(bool)

    # shorten label for clarity
    safe_label = (
        label.replace(" Ties", "")
             .replace(", ", "_")
             .replace(" ", "")
    )
    # e.g. "Max All Ties, Max Cross Ties" -> "MaxAll_MaxCross"

    # plot
    viz.plot_relative_change_panels(
        mode="Mean",
        mask=mask_pivot,
        mask_label=safe_label,
        output_folder=OUTPUT_FOLDER,
    )

viz.combine_condition_plots_vertical(
    output_folder=OUTPUT_FOLDER,
    state="initial",
    mode="All",
    base_name="relative_change_Mean",
    mask_labels=[
        "MaxAll_MaxCross",
        "MinAll_MaxCross",
        "MaxAll_MinCross",
        "MinAll_MinCross",
    ],
)