from OutputGenerator import OutputGenerator

# output_gen = OutputGenerator("Experiment_data/batch_2025-09-02_14-20-41")

output_gen = OutputGenerator("Experiment_data/batch_2025-09-04_12-20-43")

# output_gen.run_3d_visualization(trait_of_interest='Age_binary')
# output_gen.plot_heatmap(trait_of_interest='Gender_binary')
# output_gen.plot_heatmaps(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'], target_entropy=True)
# output_gen.run_3d_visualization(trait_of_interest='Gender_tertiary', target_entropy=True)
weight_choice = {'Age_tertiary': 1/3, 'Gender_tertiary': 1/3, 'EducationLevel_tertiary': 1/3}
output_gen.plot_real_entropy_boxplots(weight_choice=weight_choice, tol=1e-6)

#binary: "Experiment_data/batch_2025-08-28_15-13-05" -> "Experiment_data/batch_2025-08-28_18-00-36"
#tertiary: "Experiment_data/batch_2025-08-28_15-15-53" -> "Experiment_data/batch_2025-08-28_18-04-39"
#all categories: "Experiment_data/batch_2025-08-28_15-03-53" -> "Experiment_data/batch_2025-08-28_18-09-38"
# "->" means new batch that separates the absolute errors into separate traits