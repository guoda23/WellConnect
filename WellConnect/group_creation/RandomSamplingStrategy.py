import pandas as pd
import numpy as np
from group_creation.GroupCreationStrategy import GroupCreationStrategy

class RandomSamplingStrategy(GroupCreationStrategy):
    """
    Group creation strategy that randomly assigns individuals.
    """

    def create_groups(self):
        """
        Randomly samples individuals from the population to create groups of a specified size.

        Returns:
            pd.DataFrame: A DataFrame with individuals assigned to groups and their atrributes.
        """
        available_data = self.population_data.copy()

        groups = []

        for group_id in range(1, self.num_groups +1):
            group_members = available_data.sample(n=self.group_size, random_state=self.seed)
            available_data = available_data.drop(group_members.index)
            group_members['group_id'] = group_id
            groups.append(group_members)

        return self.group_list_to_df(groups)
