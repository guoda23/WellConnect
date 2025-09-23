from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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
                    row[attr] = 0 if value1 != value2 else 1
                    
            row['target'] = edge['weight']
            data.append(row)      

        return pd.DataFrame(data)


    def perform_group_regression(self, groups, drop_last_var = True, drop_var: str = None):
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

            if drop_var is not None:
                used_attrs = [a for a in self.attributes if a != drop_var]
                last_attr = drop_var
            elif drop_last_var:
                used_attrs = self.attributes[:-1]
                last_attr = self.attributes[-1]
            else:
                used_attrs = self.attributes
                last_attr = None

            X = regression_data[used_attrs]
            Y = regression_data['target']

            model = self.regression_model
            model.fit(X, Y)

            recovered_weights = pd.Series(model.coef_, index=used_attrs)
            
            if last_attr is not None:
                reconstructed_last = 1 - recovered_weights.sum()
                recovered_weights[last_attr] = reconstructed_last

            # Normalize to sum to 1
            reconstructed_weights = recovered_weights / recovered_weights.sum()

            recovered_weights_by_group.append({
                'group_id': group.group_id,
                **reconstructed_weights.to_dict()
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

    
    # def project_to_simplex(self, weights):
    #     """
    #     Projects a vector of weights onto the probability simplex:
    #     non-negative and summing to 1.
    #     Based on: Wang & Carreira-Perpinán (2013).
    #     """
    #     weights = np.array(weights)
    #     sorted_w = np.sort(weights)[::-1]
    #     cssv = np.cumsum(sorted_w)
    #     rho = np.nonzero(sorted_w * np.arange(1, len(weights)+1) > (cssv - 1))[0][-1]
    #     theta = (cssv[rho] - 1) / (rho + 1.0)
    #     projected = np.maximum(weights - theta, 0)
    #     return projected

    # def perform_group_regression_constrained(self, groups, drop_last_var=True):
    #     """
    #     Performs regression and then projects the recovered weights
    #     onto the probability simplex (≥0, sum=1).
    #     """
    #     recovered_weights_by_group = []

    #     for group in groups:
    #         regression_data = self.prepare_group_regression_data(group)

    #         if drop_last_var:
    #             used_attrs = self.attributes[:-1]
    #             last_attr = self.attributes[-1]
    #         else:
    #             used_attrs = self.attributes
    #             last_attr = None

    #         X = regression_data[used_attrs]
    #         Y = regression_data['target']

    #         model = self.regression_model
    #         model.fit(X, Y)

    #         recovered_weights = np.array(model.coef_)
    #         constrained_weights = self.project_to_simplex(recovered_weights)

    #         recovered_weights_by_group.append({
    #             'group_id': group.group_id,
    #             **dict(zip(used_attrs, constrained_weights))
    #         })

    #     return pd.DataFrame(recovered_weights_by_group)
    


    def constrained_regression_scipy(self, X, y):
        """
        Solve least squares regression with constraints:
        - weights >= 0
        - sum(weights) = 1
        """
        n_features = X.shape[1]

        # Loss function: squared error
        def loss_fn(w):
            return np.sum((X @ w - y) ** 2)

        # Constraints: sum of weights = 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # Bounds: weights >= 0
        bounds = [(0, None)] * n_features

        # Start with uniform weights
        init = np.ones(n_features) / n_features

        result = minimize(
            loss_fn, x0=init, bounds=bounds, constraints=constraints,
            method="SLSQP", options={'ftol':1e-9, 'maxiter':1000}
        )

        if not result.success:
            # If optimization fails, return NaNs so you can catch it
            return np.full(n_features, np.nan)
        return result.x


    def perform_group_regression_scipy(self, groups, drop_last_var=False):
        """
        Performs true constrained regression using scipy.optimize.minimize
        (≥0, sum=1 enforced during optimization).
        """
        recovered_weights_by_group = []

        for group in groups:
            regression_data = self.prepare_group_regression_data(group)

            if drop_last_var:
                used_attrs = self.attributes[:-1]
                last_attr = self.attributes[-1]
            else:
                used_attrs = self.attributes
                last_attr = None

            X = regression_data[used_attrs].to_numpy()
            Y = regression_data['target'].to_numpy()

            constrained_weights = self.constrained_regression_scipy(X, Y)

            recovered_weights_by_group.append({
                'group_id': group.group_id,
                **dict(zip(used_attrs, constrained_weights))
            })

        return pd.DataFrame(recovered_weights_by_group)