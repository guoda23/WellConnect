from group_creation.GroupCreationStrategy import GroupCreationStrategy


class GroupCreator:
    def __init__(self, strategy):
        """
        Initializes the GroupCreator with a specific group creation strategy.

        Parameters:
        - strategy (GroupCreationStrategy): An instance of a GroupCreationStrategy subclass.
        """

        if not isinstance(strategy, GroupCreationStrategy):
            raise TypeError("strategy must be an instance of GroupCreationStrategy or its subclass.")
        self.strategy = strategy


    def create_groups(self):
        """
        Creates groups using the specified strategy.

        Returns:
        - pd.DataFrame: A DataFrame with group assignments and associated attributes.
        """
        return self.strategy.create_groups()
        

