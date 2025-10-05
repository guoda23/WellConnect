
# %%

from OutputGenerator import OutputGenerator
#constrained reg:
batch_folder_deterministic = "Experiments/homophily_function_retrievability/deterministic/batch_2025-10-03_15-27-55"
# unconstrained reg:
# batch_folder_deterministic = "Experiments/homophily_function_retrievability/deterministic/batch_2025-10-04_16-30-32"


# %%
output_gen = OutputGenerator(batch_folder_deterministic, mode="deterministic")
# %%
# output_gen._load_experiment_data()
# # output_gen.plot_real_entropy_boxplots(export=False)

# #  %%
# df =output_gen.report_unreached_entropies()

# %%
output_gen.plot_heatmaps(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'], target_entropy=True,
                         dependent_variable="mean",
                         save_path="Results/homophily_f_retrievability/deterministic_heatmaps_mean.png",
                         vmax=0.45,
                         suptitle=True)
# %%
output_gen.plot_heatmaps(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'], target_entropy=True,
                         dependent_variable="std",
                         save_path="Results/homophily_f_retrievability/deterministic_heatmaps_std.png",
                         vmax=0.45,
                         suptitle=True)

# %%
output_gen.plot_combined_heatmaps(img_mean="Results/homophily_f_retrievability/deterministic_heatmaps_mean.png",
        img_std="Results/homophily_f_retrievability/deterministic_heatmaps_std.png",
        combined_out="Results/homophily_f_retrievability/deterministic_heatmaps_combined.png",
        figsize=(8, 8.5),
        show=True)


# %%
# histogram
# output_gen.plot_trait_histograms(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])
# # %%
# #collinearity
# output_gen.plot_trait_collinearity(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])
# %%
# output_gen.run_3d_visualization(trait_of_interest='Age_tertiary', target_entropy=True, seed=1)
