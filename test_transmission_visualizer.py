from TransmissionVisualizer import TransmissionVisualizer
viz = TransmissionVisualizer(
    batch_folder="Experiments/transmission/batch_2025-10-15_11-49-32"
)

viz.load_experiment_data(noise_level=0.15)

# records = viz.extract_transition_data(use_density=True)
# viz.plot_transitions(records, x_axis="density")

# viz.plot_final_state_triangles()
viz.plot_fraction_triangle_2_axes()
