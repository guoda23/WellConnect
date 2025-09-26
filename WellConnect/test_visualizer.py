from OutputGenerator import OutputGenerator


# output_gen = OutputGenerator("Experiment_data/batch_2025-09-25_15-05-11")
output_gen = OutputGenerator("Experiments/homophily_function_retrievability/stochastic/batch_2025-09-26_12-41-05")


output_gen.plot_heatmaps(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'], target_entropy=True, noise_levels=[1.5])
# output_gen.plot_noise_vs_error()



# output_gen.plot_trait_histograms(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])
# output_gen.run_3d_visualization(trait_of_interest='Age_tertiary', target_entropy=False)

# weight_choice = {'Age_tertiary': 1/3, 'Gender_tertiary': 1/3, 'EducationLevel_tertiary': 1/3}
# output_gen.plot_real_entropy_boxplots(weight_choice=weight_choice)w

