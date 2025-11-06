from TransmissionVisualizer import TransmissionVisualizer

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

# # density heatmap
# viz.plot_density_heatmap(
#     state="initial",
#     output_folder="Results/transmission/even",
#     mode="all"
# )


viz.plot_stacked_phq9_distributions(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'],
                                    output_folder="Results/transmission")


