from sklearn.linear_model import LinearRegression
import pandas as pd

class RegressionRunner:
    def __init__(self,  attributes, max_distances): #TODO: add regression model as parameter
        self.regression_model = LinearRegression()
        self.attributes = attributes
        self.max_distances = max_distances


    def prepare_group_regression_data(self, group):
        """
        Prepares data for regression analysis by calculating distances between node attributes.

        Parameters:
            group (Group): The group containing the network to analyze.

        Returns:
            pd.DataFrame: A DataFrame containing regression input data (attributes and target).
        """

        G = group.network
        data = []

        for agent1, agent2, edge in G.edges(data=True):
            row = {}

            for attr in self.attributes:
                value1 = getattr(agent1, attr, None)
                value2 = getattr(agent2, attr, None)
                max_attribute_distance = self.max_distances[attr]

                # Absolute differences for continuous attributes
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    absolute_distance = abs(value1 - value2)
                    normalized_distance = absolute_distance / max_attribute_distance
                    row[attr] = - normalized_distance

                # Binary differences for categorical attributes
                elif isinstance(value1, str) and isinstance(value2, str):
                    row[attr] = - 1 if value1 != value2 else 0 #minus because distance has negative contribution to hours
                    
            row['target'] = edge['weight']
            data.append(row)      

        return pd.DataFrame(data)


    def perform_group_regression(self, groups):
        """
        Performs regression on a list of groups to recover weights for attributes.

        Parameters:
            groups (list[Group]): List of Group objects.

        Returns:
            pd.DataFrame: A DataFrame with recovered weights for each group.
        """
        recovered_weights_by_group = []

        for group in groups:
            regression_data = self.prepare_group_regression_data(group)
            X = regression_data[self.attributes]
            Y = regression_data['target']

            model = self.regression_model
            model.fit(X, Y)

            recovered_weights = pd.Series(model.coef_, index=self.attributes)
            normalized_recovered_weights = recovered_weights / recovered_weights.sum()

            recovered_weights_by_group.append({
                'group_id': group.group_id,
                **normalized_recovered_weights.to_dict()
            })

        return pd.DataFrame(recovered_weights_by_group)
    

    def display_results(self, recovered_weights_df, true_weights=None):
        """
        Displays the regression results and optionally adds true weights for comparison.

        Parameters:
            recovered_weights_df (pd.DataFrame): DataFrame of recovered weights for each group.
            true_weights (dict, optional): True weights to include for comparison. Defaults to None.
        """
        if true_weights:
            base_weights_row = pd.DataFrame([{
                'group_id': 'True Weights',
                **true_weights
            }])
            recovered_weights_df = pd.concat([base_weights_row, recovered_weights_df], ignore_index=True)

        print("Regression Results:")
        print(recovered_weights_df)
    
