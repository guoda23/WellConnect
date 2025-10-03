import numpy as np
import random
import pandas as pd
from collections import Counter
from scipy.stats import entropy
from entities.Group import Group
from group_creation_strategies.GroupCreationStrategy import GroupCreationStrategy


class MultiTraitEntropySamplingStrategy(GroupCreationStrategy): 
    def __init__(self, agents, group_size, target_entropy, traits, num_groups=None, seed=None):
        """
        Initializes the entropy-controlled sampling strategy (for joint entropy of traits).

        Parameters:
        - agents (list[Agent]): The dataset containing Agent objects.
        - group_size (int): Number of individuals per group.
        - target_entropy (float): The target entropy for the groups.
        - traits (str): The traits to use for entropy calculation.
        - num_groups (int, optional): Number of groups to form. Defaults to the maximum possible groups.
        - seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(agents, group_size, num_groups, seed) #inherit from parent class
        self.target_entropy = target_entropy
        self.traits = traits

        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()


    def create_groups(self): #TODO: potentially make more efficient
        """
        Create groups while controlling entropy based on the specified traits.
        The first member of each group is selected randomly (seeded RNG),
        then additional members are added greedily to minimize the difference
        from the target entropy. Ties are broken randomly but reproducibly.
        """

        available_agents = self.agents.copy()
        groups = []

        for group_id in range(1, self.num_groups + 1):
            group_members = []
            current_entropy = 0.0

            while len(group_members) < self.group_size:
                if not group_members:
                    candidate = self.rng.choice(available_agents)
                else:
                    current_entropy = self.calculate_joint_entropy(group_members)
                    # Candidate selection
                    candidate = self.select_best_candidate(available_agents, group_members)

                group_members.append(candidate)
                available_agents.remove(candidate)
            
            group = Group(group_id=group_id, members=group_members)
            #update real entropy after group is filled
            group.real_entropy = self.calculate_joint_entropy(group_members)
            groups.append(group)

        return groups



    def calculate_joint_entropy(self, group_members):
        """
        Calculates Shannon entropy for multiple categorical traits.

        Parameters:
        - group_members (list[Agent]): Members of the current group.
        
        Returns:
        - float: Joint entropy value.
        """
        
        trait_combinations = [tuple(getattr(agent, trait, None) for trait in self.traits) for agent in group_members]
        combination_counts = Counter(trait_combinations)
        total_count = sum(combination_counts.values())
        probabilities = [count / total_count for count in combination_counts.values()]

        return entropy(probabilities, base=2) 
    
    
    def select_best_candidate(self, available_agents, group_members):
        """
        Selects the candidate that minimizes the entropy difference for the group.

        Parameters:
        - available_agents (list[Agent]): The remaining agents to select candidates from.
        - group_members (list[Agent]): Current members of the group.

        Returns:
        - Agent: The best candidate
        """

        best_candidates = []
        min_entropy_diff = float('inf')

        for agent in available_agents:
            # Test with the candidate
            test_group = group_members + [agent]
            test_entropy = self.calculate_joint_entropy(test_group)
            entropy_diff = abs(test_entropy - self.target_entropy)

            # Update the best candidate if this agent minimizes the entropy difference
            if entropy_diff < min_entropy_diff:
                best_candidates = [agent] #reset list
                min_entropy_diff = entropy_diff 
            elif entropy_diff == min_entropy_diff:
                best_candidates.append(agent) #add to tie list

        # break ties reproducibly using the seeded RNG
        return self.rng.choice(best_candidates)