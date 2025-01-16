import numpy as np
import pandas as pd
from scipy.stats import entropy
from group_creation_strategies.GroupCreationStrategy import GroupCreationStrategy

class EntropyControlledSamplingStrategy(GroupCreationStrategy): 
    def __init__(self, population_data, group_size, target_entropy, trait, tolerance, num_groups=None, seed=None):
        """
        Initializes the entropy-controlled sampling strategy.

        Parameters:
        - population_data (pd.DataFrame): The dataset containing individuals.
        - group_size (int): Number of individuals per group.
        - target_entropy (float): The target entropy for the groups.
        - trait (str): The trait to use for entropy calculation.
        - tolerance (float): The allowed tolerance for entropy difference.
        - num_groups (int, optional): Number of groups to form. Defaults to the maximum possible groups.
        - seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(population_data, group_size, num_groups, seed) #inherit from parent class
        self.target_entropy = target_entropy
        self.trait = trait
        self.tolerance = tolerance


    def create_groups(self): #TODO: potentially make more efficient
        if self.seed is not None:
            np.random.seed(self.seed)
        available_data = self.population_data.copy()

        groups = []

        for group_id in range(1, self.num_groups + 1):
            group_members = []
            current_entropy = 0.0

            while len(group_members) < self.group_size:
                if group_members:
                    group_trait_values = [row[self.trait] for row in group_members]
                    current_entropy = self.calculate_entropy(group_trait_values)
            
                # Stop if target entropy is achieved and group is full
                if self.should_stop_group_formation(group_members, current_entropy):
                    break

                # Candidate selection
                candidate = self.select_best_candidate(available_data, group_members)

                if candidate is not None:
                    group_members.append(candidate)
                    available_data = available_data.drop(candidate.name)

                
            # Save the group
            group_df = pd.DataFrame(group_members)
            group_df['group_id'] = group_id
            groups.append(group_df)

        return self.group_list_to_df(groups)
    

    def calculate_entropy(self, trait_values):
        """
        Calculates Shannon entropy for a given list of trait values.

        Parameters:
        - trait_values (list): A list of values for the trait being considered.

        Returns:
        - float: Shannon entropy value for the trait distribution.
        """
        value_counts = pd.Series(trait_values).value_counts(normalize=True)
        return entropy(value_counts, base=2) #Shannon entropy with base 2 log
    

    def should_stop_group_formation(self, group_members, current_entropy):
        """
        Checks if group formation should stop based on entropy and group size.

        Parameters:
        - group_members (list): Current members of the group.
        - current_entropy (float): Current Shannon entropy of the group's trait distribution.

        Returns:
        - bool: True if group formation should stop, False otherwise.
        """
        return (
            len(group_members) == self.group_size
            and abs(current_entropy - self.target_entropy) <= self.tolerance
        )
    
    
    def select_best_candidate(self, available_data, group_members):
        """
        Selects the candidate that minimizes the entropy difference for the group.

        Parameters:
        - available_data (pd.DataFrame): The remaining data to select candidates from.
        - group_members (list): Current members of the group.
        - trait (str): The trait to control entropy for.

        Returns:
        - pd.Series or None: The best candidate row, or None if no suitable candidate is found.
        """
        candidate = None
        min_entropy_diff = float('inf')

        for _, row in available_data.iterrows():
            # Test with new candidate
            test_group = group_members + [row]
            test_trait_values = [row[self.trait] for row in test_group]
            test_entropy = self.calculate_entropy(test_trait_values)
            entropy_diff = abs(test_entropy - self.target_entropy)

            # Update candidate if this row minimizes the entropy difference
            if entropy_diff < min_entropy_diff:
                candidate = row
                min_entropy_diff = entropy_diff

        return candidate