import pandas as pd
import numpy as np
import random
from entities.Group import Group
from group_creation_strategies.GroupCreationStrategy import GroupCreationStrategy

class RandomSamplingStrategy(GroupCreationStrategy):
    """
    Group creation strategy that randomly assigns individuals.
    """

    def create_groups(self):
        """
        Randomly samples individuals from the population to create groups of a specified size.

        Returns:
        - list[Group]: A list of Group objects with randomly assigned members.
        """
        available_agents = self.agents.copy()
        random.shuffle(available_agents)

        groups = []

        for group_id in range(1, self.num_groups +1):
            group_members = available_agents[:self.group_size]
            group = Group(group_id=group_id, members=group_members)
            groups.append(group)

            if not available_agents:
                break

        return groups
