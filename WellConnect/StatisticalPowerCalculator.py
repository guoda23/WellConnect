import numpy as np
import pandas as pd

#TODO: assess how statistically sound this approach is (This is more of a placeholder)
# N.B. coverage probability seems to make the most sense

import numpy as np
import pandas as pd

class StatisticalPowerCalculator:
    def __init__(self, recovered_weights_df, true_weights, n_bootstraps=1000, confidence_level=0.95):
        """
        Initializes the StatisticalPowerCalculator.

        Parameters:
        - recovered_weights_df (pd.DataFrame): DataFrame containing recovered weights for each group.
        - true_weights (dict): Dictionary of true weights for each attribute.
        - n_bootstraps (int, optional): Number of bootstrap samples to use. Defaults to 1000.
        - confidence_level (float, optional): Confidence level for intervals. Defaults to 0.95.
        """
        self.recovered_weights_df = recovered_weights_df
        self.true_weights = true_weights
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level


    def absolute_error(self, NaN_penalty = 1):
        total_differences_per_group = {}
        
        for group_id, row in self.recovered_weights_df.iterrows():
            group_difference = 0

            for attribute, true_value in self.true_weights.items(): #for every group, add up the error for every attribute column
                recovered_value = row[attribute]

                if pd.isna(recovered_value):
                    group_difference += NaN_penalty
                else:
                    group_difference += abs(recovered_value - true_value)
        
            total_differences_per_group[group_id] = group_difference 

        return total_differences_per_group


    def bootstrap_confidence_interval(self, data):
        """
        Computes the bootstrap confidence interval for the given data.

        Parameters:
        - data (np.ndarray): Array of data to calculate confidence intervals.

        Returns:
        - tuple: Lower and upper bounds of the confidence interval.
        """
        bootstrap_samples = np.random.choice(data, size=(self.n_bootstraps, len(data)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        lower_bound = np.percentile(bootstrap_means, (1 - self.confidence_level) / 2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 + self.confidence_level) / 2 * 100)
        return lower_bound, upper_bound


    def calculate_coverage_probability(self, data, true_value):
        """
        Calculates the coverage probability for the true value given bootstrap samples.

        Parameters:
        - data (np.ndarray): Array of data to calculate coverage probability.
        - true_value (float): The true value to check inclusion in confidence intervals.

        Returns:
        - float: Coverage probability.
        """
        lower_bound, upper_bound = self.bootstrap_confidence_interval(data)
        in_interval = (lower_bound <= true_value) & (upper_bound >= true_value)
        return np.mean(in_interval)


    def calculate_bias(self, data, true_value):
        """
        Calculates the bias of the recovered weights.

        Parameters:
        - data (np.ndarray): Array of data to calculate bias.
        - true_value (float): The true value for the bias calculation.

        Returns:
        - float: Bias of the recovered weights.
        """
        bootstrap_samples = np.random.choice(data, size=(self.n_bootstraps, len(data)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        return np.mean(bootstrap_means) - true_value


    def calculate_mse(self, data, true_value):
        """
        Calculates the mean squared error (MSE) of the recovered weights.

        Parameters:
        - data (np.ndarray): Array of data to calculate MSE.
        - true_value (float): The true value for MSE calculation.

        Returns:
        - float: Mean squared error of the recovered weights.
        """
        bootstrap_samples = np.random.choice(data, size=(self.n_bootstraps, len(data)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        return np.mean((bootstrap_means - true_value) ** 2)


    def evaluate_predictive_power(self, attribute): #TODO: polish this for thesis goals. Attribute?
        """
        Evaluates predictive power for a specific attribute.

        Parameters:
        - attribute (str): Name of the attribute to evaluate.

        Returns:
        - dict: Results including coverage probability, bias, MSE, and confidence interval.
        """
        data = self.recovered_weights_df[attribute].values
        true_value = self.true_weights.get(attribute)

        if true_value is None:
            raise ValueError(f"True weight for attribute '{attribute}' not found.")

        lower_bound, upper_bound = self.bootstrap_confidence_interval(data)
        coverage = self.calculate_coverage_probability(data, true_value)
        bias = self.calculate_bias(data, true_value)
        mse = self.calculate_mse(data, true_value)

        return {
            "attribute": attribute,
            "true_value": round(true_value, 3) if isinstance(true_value, (int, float)) else true_value,
            "coverage_probability": round(coverage, 3),
            "bias": round(bias, 3),
            "mse": round(mse, 3),
            "confidence_interval": (round(lower_bound, 3), round(upper_bound, 3)),
            "absolute_error": self.absolute_error()
        }

