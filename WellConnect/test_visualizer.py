from OutputGenerator import OutputGenerator

# output_gen = OutputGenerator("Experiment_data/batch_2025-09-02_14-20-41")

output_gen = OutputGenerator("Experiment_data/batch_2025-09-05_12-01-19")

# output_gen.run_3d_visualization(trait_of_interest='Age_binary')
# output_gen.plot_heatmap(trait_of_interest='Gender_binary')
output_gen.plot_heatmaps(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'], target_entropy=False)
# output_gen.run_3d_visualization(trait_of_interest='Gender_tertiary', target_entropy=True)
# weight_choice = {'Age_tertiary': 1/3, 'Gender_tertiary': 1/3, 'EducationLevel_tertiary': 1/3}
# output_gen.plot_real_entropy_boxplots(weight_choice=weight_choice)

