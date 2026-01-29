import numpy as np
import pandas as pd

class StatisticalPowerCalculator:
    def __init__(self, recovered_weights_df, true_weights):
        """
        Wrapper around statistical evaluation functions.
        
        Parameters:
        - recovered_weights_df (pd.DataFrame): DataFrame with recovered weights per group.
        - true_weights (dict): True weights for each attribute.
        """
        self.recovered_weights_df = recovered_weights_df
        self.true_weights = true_weights

    # --- error metrics ---
    def absolute_error(self, attribute, nan_penalty=1, anomaly_penalty=1):
        errors_per_group = {}

        for group_id, row in self.recovered_weights_df.iterrows():
            recovered_value = row[attribute]
            true_value = self.true_weights[attribute]

            if pd.isna(recovered_value):
                errors_per_group[group_id] = nan_penalty
            elif recovered_value < 0 or recovered_value > 1:
                errors_per_group[group_id] = anomaly_penalty
            else:
                errors_per_group[group_id] = abs(recovered_value - true_value)

        return errors_per_group


    def combined_absolute_error(self, nan_penalty=1, anomaly_penalty=1):
        totals = {}
        for group_id, row in self.recovered_weights_df.iterrows():
            diff = 0
            for attribute, true_value in self.true_weights.items():
                recovered_value = row[attribute]
                if pd.isna(recovered_value):
                    diff += nan_penalty
                elif recovered_value < 0 or recovered_value > 1:
                    diff += anomaly_penalty
                else:
                    diff += abs(recovered_value - true_value)
            totals[group_id] = diff
        return totals

    #TODO: add more error metrics (MSE, bias, coverage probability, CI width)

    # --- convenience wrapper ---
    def evaluate_predictive_power_trait_specific(self, attributes, **kwargs):
        """
        Trait-specific evaluation (per group).
        Returns results per attribute, with group-level detail.
        """
        measure_dict = {}

        for attribute in attributes:
            data = self.recovered_weights_df[attribute].values
            true_value = self.true_weights.get(attribute)

            measure_dict[attribute] = {
                "attribute": attribute,
                "true_value": round(true_value, 3) if isinstance(true_value, (int, float)) else true_value,
                "absolute_error": self.absolute_error(attribute, **kwargs),
            }

        return measure_dict


    def evaluate_predictive_power_combined_attributes(self):
        """
        Combined evaluation across all attributes (per group).
        Returns one combined absolute error per group across all traits,
        alongside placeholder statistics (currently computed on all data pooled).
        """
        measure_dict = {}

        for group_id, error_value in self.combined_absolute_error().items():
            measure_dict[group_id] = {
                "group_id": group_id,
                "combined_absolute_error": error_value
            }

        return measure_dict
