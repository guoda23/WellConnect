from networkx import density
from TransmissionVisualizer import TransmissionVisualizer
import pandas as pd

viz = TransmissionVisualizer(
    batch_folder = "Experiments/transmission/batch_2025-10-31_17-50-13" # 0.7 age, 0.2 edu, 0.1 gender
    # batch_folder = "Experiments/transmission/batch_2025-10-31_19-10-40" # 1/3 age, 1/3 edu, 1/3 gender

)
viz.load_experiment_data(noise_level=0.15)


#N.B. adjust vmax beforehand if needed
# viz.plot_relative_change_panels(mode="Mean", output_folder="Results/transmission/even")
# viz.plot_relative_change_panels(mode="Std", output_folder="Results/transmission/even")
# viz.plot_relative_change_panels(mode="Counts", output_folder="Results/transmission/even")


# viz.combine_relative_change_plots("Results/transmission/")

# viz.plot_group_count_distribution(output_folder="Results/transmission/")

# density heatmap
# viz.plot_density_heatmap(
#     state="initial",
#     output_folder="Results/transmission", #TODO: add /even for even weights
#     mode="All",
#     # vmax_mean=0.85,
#     # vmax_std=0.85,
#     # vmin_mean=0.0,
#     # vmin_std=0.0
# )

# viz.plot_density_heatmap(
#     state="initial",
#     output_folder="Results/transmission", #TODO: add /even for even weights
#     mode="Cross",
#     # vmax_mean=0.425,
#     # vmax_std=0.425,
#     # vmin_mean=0.0,
#     # vmin_std=0.0
# )



# viz.combine_density_heatmaps("Results/transmission/")


# viz.plot_density_percentiles_2x2(
#     state="initial",
#     output_folder="Results/transmission"
# )

grouped, conds = viz.plot_density_percentiles_2x2(state="initial")

# build boolean pivot mask
mask_series = conds["Max All Ties, Max Cross Ties"]
mask_pivot = pd.pivot_table(
    grouped.assign(ConditionMet=mask_series),
    values="ConditionMet",
    index="Mo_rounded",
    columns="S_rounded",
    aggfunc=lambda x: any(x),   # boolean aggregation
)

# make sure dtype is boolean
mask_pivot = mask_pivot.astype(bool)


viz.plot_relative_change_panels(
    mode="Mean",
    mask=mask_pivot,
    mask_label="MaxAll_MaxCross",
    output_folder="Results/transmission"
)


# viz.combine_condition_plots_vertical(
#     output_folder="Results/transmission",
#     state="initial",
#     mode="All",
#     base_name="relative_change_Mean",
#     mask_labels=[
#         "MaxAll_MaxCross",
#         "MinAll_MaxCross",
#         "MaxAll_MinCross",
#         "MinAll_MinCross",
#     ],
# )


# viz.plot_stacked_phq9_distributions(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'],
#                                     output_folder="Results/transmission")


