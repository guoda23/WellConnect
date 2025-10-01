from abc import ABC, abstractmethod
import pandas as pd

class GroupCreationStrategy(ABC):
    """
    Abstract base class for all group creation strategies.
    """

    def __init__(self, agents, group_size, num_groups = None, seed = None):
        """
        Initializes the strategy with shared attributes.

        Parameters:
        - agents (list[Agent]): The dataset containing Agent objects.
        - group_size (int): Number of individuals per group.
        - num_groups (int, optional): Number of groups to form. Defaults to the maximum possible groups.
        - seed (int, optional): Random seed for reproducibility.

        Raises:
        - ValueError: If the group size and number of groups exceed the population size.
        """
        
        self.agents = agents
        self.group_size = group_size
        self.seed = seed
        
        # Default num_groups to the maximum possible groups
        if num_groups is None:
            self.num_groups = len(agents) // group_size
        else:
            self.num_groups = num_groups

        self.validate_group_size_for_population() # Check if large enough population for number of groups


    def validate_group_size_for_population(self):
        """
        Validates that the group size and number of groups fit within the population size.

        Raises:
        - ValueError: If the population size is insufficient.
        """

        if self.group_size * self.num_groups > len(self.agents):
            raise ValueError("Not enough data to create the specified number of groups.")
        

    @abstractmethod 
    def create_groups(self, **kwargs):
        """
        Abstract method to create groups.

        Parameters:
        - kwargs: Additional parameters for specific strategies.

        Returns:
        - list[Group]: A list of Group objects.
        """
        pass


    # def group_list_to_df(self, groups): #TODO: remove after switching to group/agent class
        # """
        # Converts a group list of DataFrames into a single DataFrame with a group_id attribute.

        # Parameters:
        # - groups (list): A list of DataFrames.

        # Returns:
        # - pd.DataFrame: A DataFrame with group assignments and associated attributes.
        # """

        # all_groups_df = pd.concat(groups, axis=0).reset_index(drop=True)
        # # Reorder columns to place 'group_id' first
        # columns = ['group_id'] + [col for col in all_groups_df.columns if col != 'group_id']
        # all_groups_df = all_groups_df[columns]

        # return all_groups_df
    
