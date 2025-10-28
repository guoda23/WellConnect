from TransmissionVisualizer import TransmissionVisualizer
from SupercohortTransmissionVisualizer import SupercohortTransmissionVisualizer

viz = TransmissionVisualizer(
    # batch_folder="Experiments/transmission/batch_2025-10-18_16-53-22" #Hill's (50 steps!)
    # batch_folder="Experiments/transmission/batch_2025-10-18_17-22-16" #van der Ende's (50 steps!)
    # batch_folder = "Experiments/transmission/batch_2025-10-20_20-27-17" # new hill with transition log (20 steps!)
    # batch_folder = "Experiments/transmission/batch_2025-10-20_20-51-09" # new van der Ende with transition log (20 steps!)
    # batch_folder = "Experiments/transmission/batch_2025-10-21_14-54-02"
    # batch_folder = "Experiments/transmission/batch_2025-10-21_15-07-16" # longer runs (500 steps, group of 30)
    # batch_folder = "Experiments/transmission/batch_2025-10-22_15-40-55" # 30 seeds
    # batch_folder = "Experiments/transmission/batch_2025-10-23_12-15-02" # fixed entropy (70 seeds)
    # batch_folder = "Experiments/transmission/batch_2025-10-28_11-04-35" 
    batch_folder = "Experiments/transmission/batch_2025-10-28_13-39-47"

)

viz.load_experiment_data(noise_level=0.15)

# viz.plot_relative_change_panels(mode="raw", normalize=True)




# density heatmap
viz.plot_density_heatmap(state="final")
# density heatmap (line across)
# viz.plot_density_triangle_2_axes(state="initial")

