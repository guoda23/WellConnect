from OutputGenerator import OutputGenerator


# output_gen = OutputGenerator("Experiment_data/batch_2025-09-25_15-05-11")
output_gen = OutputGenerator("Experiments/homophily_function_retrievability/deterministic/batch_test", mode="deterministic")


output_gen.plot_heatmaps(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'], target_entropy=True, dependent_variable="mean")
# output_gen.plot_noise_vs_error()

#histograms
# output_gen.plot_trait_histograms(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])

#collinearity
# output_gen.plot_trait_collinearity(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])


# output_gen.run_3d_visualization(trait_of_interest='Age_tertiary', target_entropy=False) #TODO: fix this for the new experiment structure


# output_gen.plot_real_entropy_boxplots(weight_choice=weight_choice)w

