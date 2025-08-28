import numpy as np
import random
import pandas as pd
from collections import Counter
from scipy.stats import entropy
from entities.Group import Group
from group_creation_strategies.GroupCreationStrategy import GroupCreationStrategy


class EntropyControlledSamplingStrategy(GroupCreationStrategy): 
    def __init__(self, agents, group_size, target_entropy, trait, tolerance, num_groups=None, seed=None):
        """
        Initializes the entropy-controlled sampling strategy (for a single chosen trait).

        Parameters:
        - agents (list[Agent]): The dataset containing Agent objects.
        - group_size (int): Number of individuals per group.
        - target_entropy (float): The target entropy for the groups.
        - trait (str): The trait to use for entropy calculation.
        - tolerance (float): The allowed tolerance for entropy difference.
        - num_groups (int, optional): Number of groups to form. Defaults to the maximum possible groups.
        - seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(agents, group_size, num_groups, seed) #inherit from parent class
        self.target_entropy = target_entropy
        self.trait = trait
        self.tolerance = tolerance

        if seed is not None:
            random.seed(seed)


    def create_groups(self): #TODO: potentially make more efficient
        """
        Create groups while controlling entropy based on the specified trait.

        Returns:
        - list[Group]: A list of Group objects.
        """

        available_agents = self.agents.copy()
        groups = []

        for group_id in range(1, self.num_groups + 1):
            group_members = []
            current_entropy = 0.0

            while len(group_members) < self.group_size:
                if group_members:
                    group_trait_values = [getattr(agent, self.trait, None) for agent in group_members]
                    current_entropy = self.calculate_entropy(group_trait_values)
            
                # Stop if target entropy is achieved and group is full
                if self.should_stop_group_formation(group_members, current_entropy):
                    break

                # Candidate selection
                candidate = self.select_best_candidate(available_agents, group_members)

                if candidate is not None:
                    group_members.append(candidate)
                    available_agents.remove(candidate)
            
            group = Group(group_id=group_id, members=group_members)
            groups.append(group)
                
            if not available_agents:
                break

        return groups


    def calculate_entropy(self, trait_values):
        """
        Calculates Shannon entropy for a given list of trait values.

        Parameters:
        - trait_values (list): A list of values for the trait being considered.

        Returns:
        - float: Shannon entropy value for the trait distribution.
        """
        # Count the occurrences of each value in the trait values
        value_counts = Counter(trait_values)
        # Normalize the counts to probabilities
        probabilities = [count / len(trait_values) for count in value_counts.values()]
        return entropy(probabilities, base=2) #Shannon entropy with base 2 log
    

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
    
    
    def select_best_candidate(self, available_agents, group_members):
        """
        Selects the candidate that minimizes the entropy difference for the group.

        Parameters:
        - available_agents (list[Agent]): The remaining agents to select candidates from.
        - group_members (list[Agent]): Current members of the group.

        Returns:
        - Agent: The best candidate
        """

        best_candidate = None
        min_entropy_diff = float('inf')

        for agent in available_agents:
            # Test with the candidate
            test_group = group_members + [agent]
            test_trait_values = [getattr(a, self.trait, None) for a in test_group]
            test_entropy = self.calculate_entropy(test_trait_values)
            entropy_diff = abs(test_entropy - self.target_entropy)

            # Update the best candidate if this agent minimizes the entropy difference
            if entropy_diff < min_entropy_diff:
                best_candidate = agent
                min_entropy_diff = entropy_diff

        return best_candidate