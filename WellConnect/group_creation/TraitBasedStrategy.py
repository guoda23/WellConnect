import numpy as np
import pandas as pd
from group_creation.GroupCreationStrategy import GroupCreationStrategy

class TraitBasedStrategy(GroupCreationStrategy):
    def __init__(self, population_data, group_size, trait, secondary_trait=None, num_groups=None, seed=None):
        super().__init__(population_data, group_size, num_groups, seed) #inherit from parent class
        self.trait = trait
        self.secondary_trait = secondary_trait

    def create_groups(self): #TODO: break down into smaller methods
        """
        Creates groups with a bias towards individuals having the same trait value (or combination of traits value).

        Returns:
            pd.DataFrame: A DataFrame with the following columns:
            - `group_id`: Group membership ID for each individual.
            - All other columns from the input `population_data`.
        """

        if self.seed is not None:
            np.random.seed(self.seed)
        available_data = self.population_data.copy()
        
        groups = []

        for group_id in range(1, self.num_groups + 1): #group IDs start at 1
            group_members = []

            anchor = available_data.sample(1, random_state = self.seed)
            anchor_value = anchor[self.trait].values[0]
            group_members.append(anchor)
            available_data = available_data.drop(anchor.index)

            #add similar individuals
            similar_trait_data = available_data[available_data[self.trait] == anchor_value]

            while len(group_members) < self.group_size and not available_data.empty:
                if not similar_trait_data.empty: # Select an indiv with the same trait value
                    next_individual = similar_trait_data.sample(1) 
                elif self.secondary_trait: # If a secondary trait is specified, select based on that
                    current_secondary_values = pd.concat(group_members)[self.secondary_trait]
                    majority_secondary_value = current_secondary_values.mode()[0]  # Find group majority
                    secondary_trait_data = available_data[available_data[self.secondary_trait] == majority_secondary_value]
                    if not secondary_trait_data.empty:
                        next_individual = secondary_trait_data.sample(1)
                    else: # Fallback to random
                        next_individual = available_data.sample(1)  
                else: # Final Fallback: Randomly select if no matches remain
                    next_individual = available_data.sample(1)      
                    next_individual = available_data.sample(1) 

                group_members.append(next_individual)
                available_data = available_data.drop(next_individual.index)
                similar_trait_data = available_data[available_data[self.trait] == anchor_value]


            # Combine group members into a DataFrame and save the group
            group_df = pd.concat(group_members, axis=0)
            group_df['group_id'] = group_id
            groups.append(group_df)

        return self.group_list_to_df(groups)