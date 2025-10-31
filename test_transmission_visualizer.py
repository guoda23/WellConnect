from TransmissionVisualizer import TransmissionVisualizer

viz = TransmissionVisualizer(
    batch_folder = "Experiments/transmission/batch_2025-10-28_13-39-47"

)

viz.load_experiment_data(noise_level=0.15)
# # Step 1: Generate and save both
# viz.plot_relative_change_panels(mode="Mean", output_folder="Results/transmission")
# viz.plot_relative_change_panels(mode="Std", output_folder="Results/transmission")
# viz.plot_relative_change_panels(mode="Counts", output_folder="Results/transmission")


# # Step 2: Combine them
# viz.combine_relative_change_plots("Results/transmission")

# density heatmap
# viz.plot_density_heatmap(
#     state="initial",
#     output_folder="Results/transmission"w
# )


viz.plot_stacked_phq9_distributions(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'],
                                    output_folder="Results/transmission")


