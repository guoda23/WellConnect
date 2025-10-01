
# %%

from OutputGenerator import OutputGenerator
batch_folder_deterministic = "Experiments/homophily_function_retrievability/deterministic/batch_2025-09-26_22-47-38"
batch_folder_stochastic = "Experiments/homophily_function_retrievability/stochastic/batch_2025-09-27_00-14-10"

#
# output_gen = OutputGenerator("Experiments/homophily_function_retrievability/stochastic/batch_2025-09-27_00-14-10", mode="stochastic")

# output_gen._load_experiment_data(noise_level=0.30)
# output_gen.plot_real_entropy_boxplots(export=False)


# %%
output_gen = OutputGenerator("Experiments/homophily_function_retrievability/deterministic/batch_2025-09-26_22-47-38", mode="deterministic")

output_gen._load_experiment_data()
# output_gen.plot_real_entropy_boxplots(export=False)

# %%
df =output_gen.report_unreached_entropies()

#%%


# output_gen.plot_heatmaps(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'], target_entropy=False,
#                          dependent_variable="mean", vmax=0.45, save_path="Results/homophily_f_retrievability/heatmap_noise_0.30.png")

# output_gen.build_noise_error_summary(batch_folder="Experiments/homophily_function_retrievability/stochastic/batch_2025-09-27_00-14-10")

# df_clean = output_gen.clean_noise_summary(
#     csv_in="Results/noise_error_summary.csv",
#     csv_out="Results/noise_error_summary_clean.csv"
# )

# output_gen.plot_noise_vs_error(cache_path="Results/noise_error_summary_clean.csv")
#histogram
# output_gen.plot_trait_histograms(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])

#collinearity
# output_gen.plot_trait_collinearity(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])


# output_gen.run_3d_visualization(trait_of_interest='Age_tertiary', target_entropy=False) #TODO: fix this for the new experiment structure


# noise by error plot
# output_gen.build_noise_error_summary(batch_folder=batch_folder_stochastic)
# output_gen.clean_noise_summary(batch_folder=batch_folder_stochastic)
# output_gen.plot_noise_vs_error(batch_folder=batch_folder_stochastic)




