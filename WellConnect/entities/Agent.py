class Agent:
    def __init__(self, agent_id, attribute_dict):
        """
        Initialize an Agent object.

        Parameters:
        - agent_id (int): Unique identifier for the agent.
        - attributes (dict): Dictionary of filtered attributes for the agent.
        """
        self.agent_id = agent_id

        for key, value in attribute_dict.items():
            setattr(self, key, value) #dynamically create attributes


    def __eq__(self, other):
        """
        Equality check for Agent objects.
        Agents are considered equal if their agent_id is the same.
        """
        if isinstance(other, Agent):
            return self.agent_id == other.agent_id
        return False


    def __hash__(self):
        """
        Hash function for Agent objects.
        The hash is based on the unique agent_id.
        """
        return hash(self.agent_id)


    def __repr__(self):
        return f"Agent(id={self.agent_id}, attributes={self.__dict__})"