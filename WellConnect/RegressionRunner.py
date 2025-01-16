from sklearn.linear_model import LinearRegression
import pandas as pd

class RegressionRunner:
    def __init__(self,  attributes, max_distances): #TODO: add regression model as parameter
        self.regression_model = LinearRegression()
        self.attributes = attributes
        self.max_distances = max_distances


    def prepare_group_regression_data(self, G):
        """
        Prepares data for regression analysis by calculating distances between node attributes.

        Parameters:
            G (networkx.Graph): The graph containing nodes and edges.

        Returns:
            pd.DataFrame: A DataFrame containing regression input data (attributes and target).
        """

        data = []

        for node1, node2, edge in G.edges(data=True):
            row = {}

            for attr in self.attributes:
                value1 = G.nodes[node1].get(attr, 0)
                value2 = G.nodes[node2].get(attr, 0)
                max_attribute_distance = self.max_distances[attr]

                # Absolute differences for continuous attributes
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)): #TODO: add handling o NA values
                    absolute_distance = abs(value1 - value2)
                    normalized_distance = absolute_distance / max_attribute_distance
                    row[attr] = - normalized_distance

                # Binary differences for categorical attributes
                elif isinstance(value1, str) and isinstance(value2, str):
                    row[attr] = - 1 if value1 != value2 else 0 #minus because distance has negative contribution to hours
                    
            row['target'] = edge['weight']
            data.append(row)      

        return pd.DataFrame(data)


    def perform_group_regression(self, group_graphs):
        """
        Performs regression on group graphs to recover weights for attributes based on predicted weights.

        Parameters:
            group_graphs (dict): Dictionary of graphs, keyed by group ID.

        Returns:
            pd.DataFrame: A DataFrame with recovered weights for each group.
        """
        recovered_weights_by_group = []

        for group_id, G in group_graphs.items():
            regression_data = self.prepare_group_regression_data(G)
            X = regression_data[self.attributes]
            Y = regression_data['target']

            model = self.regression_model
            model.fit(X, Y)

            recovered_weights = pd.Series(model.coef_, index=self.attributes)
            normalized_recovered_weights = recovered_weights / recovered_weights.sum()

            recovered_weights_by_group.append({
                'group_id': group_id,
                **normalized_recovered_weights.to_dict()
            })

        recovered_weights_df = pd.DataFrame(recovered_weights_by_group)
        return recovered_weights_df
    

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
    
