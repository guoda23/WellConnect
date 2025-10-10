
# %%

from OutputGenerator import OutputGenerator
#constrained reg:
batch_folder_stochastic = "Experiments/homophily_function_retrievability/stochastic/batch_2025-10-04_00-43-50"
#constrained reg trait entropy 1.761 val:
# batch_folder_stochastic = "Experiments/homophily_function_retrievability/stochastic/batch_2025-10-06_12-33-43"
#unconstrained reg:
# batch_folder_stochastic = "Experiments/homophily_function_retrievability/stochastic/batch_2025-10-04_22-52-24"

# %%
output_gen = OutputGenerator(batch_folder_stochastic, mode="stochastic")

# %%
# output_gen.run_3d_visualization(trait_of_interest='Age_tertiary', target_entropy=True, seed=1, noise_level=0.30)

# %%
for noise_level in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        output_gen._load_experiment_data(noise_level=noise_level)
        custom_trait_labels = {
        'Gender_tertiary': f'σ = {noise_level:.2f}',}

        # %%
        output_gen.plot_heatmaps(traits=['Gender_tertiary'], target_entropy=True,
                                dependent_variable="mean", 
                                vmax=0.45,
                                save_path=f"Results/homophily_f_retrievability/stochastic_heatmap_noise_{noise_level}.png",
                                suptitle=False, custom_trait_labels=custom_trait_labels, dpi=300,
                                export=False)


#  %%
# # histogram 
# output_gen.plot_trait_histograms(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])
# # %%
#collinearity
# output_gen.plot_trait_collinearity(traits=['Gender_tertiary', 'Age_tertiary', 'EducationLevel_tertiary'])
# %%

# output_gen.run_3d_visualization(trait_of_interest='Age_tertiary', target_entropy=False) #TODO: fix this for the new experiment structure

# %%
# noise by error plot
# output_gen.build_noise_error_summary(batch_folder=batch_folder_stochastic)
# output_gen.clean_noise_summary(batch_folder=batch_folder_stochastic)

# #  %%
# output_gen.plot_noise_vs_error(batch_folder=batch_folder_stochastic, save_path="Results/homophily_f_retrievability/unconstrained_stochastic_noise_vs_error.png")
# %%

