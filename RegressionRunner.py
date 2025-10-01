import pandas as pd
import numpy as np
from regression_methods import REGRESSION_FUNCTIONS


class RegressionRunner:
    def __init__(self, attributes, max_distances, regression_type="constrained"):
        """
        Parameters:
            attributes (list[str]): Which traits to use.
            max_distances (dict): Max distances per attribute for normalization.
            regression_type (str): Type of regression to perform.
        """
        self.attributes = attributes
        self.max_distances = max_distances

        # Store the actual function under self.regression_function
        if isinstance(regression_type, str):
            if regression_type not in REGRESSION_FUNCTIONS:
                raise ValueError(
                    f"Unknown regression: {regression_type}. "
                    f"Available: {list(REGRESSION_FUNCTIONS.keys())}"
                )
            self.regression_function = REGRESSION_FUNCTIONS[regression_type]


    def prepare_group_regression_data(self, group):
        """
        Prepares regression input data for a group by calculating pairwise distances.

        Returns:
            pd.DataFrame with columns = attributes + ["target"]
        """
        G = group.network
        data = []

        for agent1, agent2, edge in G.edges(data=True):
            row = {}
            for attr in self.attributes:
                value1 = getattr(agent1, attr, None)
                value2 = getattr(agent2, attr, None)
                max_attribute_distance = self.max_distances[attr]

                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    absolute_distance = abs(value1 - value2)
                    normalized_distance = absolute_distance / max_attribute_distance
                    row[attr] = -normalized_distance

                elif isinstance(value1, str) and isinstance(value2, str):
                    row[attr] = 1 if value1 == value2 else 0

            row['target'] = edge['weight']
            data.append(row)

        return pd.DataFrame(data)


    def perform_group_regression(self, groups, drop_last_var=False, drop_var=None):
        """
        Runs regression (using regression_function) on each group and recovers weights.
        """
        recovered_weights_by_group = []

        for group in groups:
            regression_data = self.prepare_group_regression_data(group)

            if drop_var is not None:
                used_attrs = [a for a in self.attributes if a != drop_var]
                last_attr = drop_var
            elif drop_last_var:
                used_attrs = self.attributes[:-1]
                last_attr = self.attributes[-1]
            else:
                used_attrs = self.attributes
                last_attr = None

            X = regression_data[used_attrs].to_numpy()
            y = regression_data['target'].to_numpy()

            weights = self.regression_function(X, y)
            recovered_weights = pd.Series(weights, index=used_attrs)

            if last_attr is not None:
                reconstructed_last = 1 - recovered_weights.sum()
                recovered_weights[last_attr] = reconstructed_last

            reconstructed_weights = recovered_weights / recovered_weights.sum()

            recovered_weights_by_group.append({
                "group_id": group.group_id,
                **reconstructed_weights.to_dict()
            })

        return pd.DataFrame(recovered_weights_by_group)

    def display_results(self, recovered_weights_df, true_weights=None):
        """
        Display regression results, optionally including true weights for comparison.
        """
        if true_weights:
            base_row = pd.DataFrame([{
                "group_id": "True Weights",
                **true_weights
            }])
            recovered_weights_df = pd.concat([base_row, recovered_weights_df], ignore_index=True)

        print("Regression Results:")
        print(recovered_weights_df)
