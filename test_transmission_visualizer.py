from TransmissionVisualizer import TransmissionVisualizer
viz = TransmissionVisualizer(
    # batch_folder="Experiments/transmission/batch_2025-10-18_16-53-22" #Hill's
    # batch_folder="Experiments/transmission/batch_2025-10-18_17-22-16" #van der Ende's
    # batch_folder = "Experiments/transmission/batch_2025-10-20_20-27-17" # new hill with transition log
    batch_folder = "Experiments/transmission/batch_2025-10-20_20-51-09" # new van der Ende with transition log
)

viz.load_experiment_data(noise_level=0.15)

# line graph: number of transitions (each category) across network density
# records = viz.extract_transition_data(use_density=True)
# viz.plot_transitions(records, x_axis="density")


# final count heatmaps:
viz.plot_final_state_heatmap_panels()
# final count across 3 axis triangle (dots)
# viz.plot_final_state_triangles_3_axes() 
# final fraction across 2 axis triangle (dots)
# viz.plot_fraction_triangles_2_axes()



# density heatmap
# viz.plot_density_heatmap(state="final")
# density heatmap (line across)
viz.plot_density_triangle_2_axes(state="initial")

