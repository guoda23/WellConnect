import networkx as nx

class Group:
    def __init__(self, group_id, members=None):
        """
        Initialize a Group object.

        Parameters:
        - group_id (int): Unique identifier for the group.
        - members (list[Agent], optional): List of Agent objects that belong to the group. Defaults to an empty list.
        - group_size (int, optional): The expected size of the group. Defaults to None.
        """

        self.group_id = group_id #NOTE that this only unique within one cohort rn!
        self.members = members if members else []
        self.group_size = len(self.members)
        self.network = None


    def add_member(self, agent):
        """
        Add an Agent to the group.

        Parameters:
        - agent (Agent): The Agent object to add to the group.
        """
        self.members.append(agent)


    def remove_member(self, agent):
        """
        Remove an Agent from the group.

        Parameters:
        - agent (Agent): The Agent object to remove from the group.
        """
        self.members.remove(agent)


    def get_member_ids(self):
        """
        Get a list of agent IDs for all members in the group.

        Returns:
        - list[int]: List of agent IDs.
        """
        return [member.agent_id for member in self.members]
    

    def get_member_attributes(self, attribute):
        """
        Get the values of a specific attribute for all members in the group.

        Parameters:
        - attribute (str): The attribute name to retrieve.

        Returns:
        - list[Any]: List of attribute values.
        """
        return [getattr(member, attribute, None) for member in self.members]
    

    # def create_group_graph(self):
    #     """
    #     Builds a NetworkX graph for this group.

    #     The graph is a complete graph where nodes represent agents, and node attributes
    #     are populated from the Agent objects.

    #     Returns:
    #     - networkx.Graph: The created network for the group.
    #     """

    #     G = nx.complete_graph(self.group_size)

    #     # Relabel nodes with person IDs
    #     mapping = {i: agent.agent_id for i, agent in enumerate(self.members)}
    #     G = nx.relabel_nodes(G, mapping)

    #     # Assign node attributes
    #     for agent in self.members:
    #         G.nodes[agent.agent_id].update(agent.__dict__)

    #     self.network = G

    #     return G

    def create_group_graph(self):
        """
        Builds a NetworkX graph for this group.

        The graph is a complete graph where nodes represent agents, and node attributes
        are populated from the Agent objects.

        Returns:
        - networkx.Graph: The created network for the group.
        """

        G = nx.Graph()

        for agent in self.members:
            G.add_node(agent)

        # Assign node attributes
        for i, agent1 in enumerate(self.members):
            for agent2 in self.members[i+1:]:
                G.add_edge(agent1, agent2)

        self.network = G
        return G


    def __repr__(self):
        return f"Group(id={self.group_id}, size={len(self.members)}, members={[member.agent_id for member in self.members]})"

    