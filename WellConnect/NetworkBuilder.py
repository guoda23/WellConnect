import networkx as nx

class NetworkBuilder: #TODO: remove class (now integrated in Group class)
    def __init__(self, groups):
        """
        Initializes the NetworkBuilder.

        Parameters:
        - groups (list[Group]): List of Group objects.
        - group_size (int): Number of individuals per group.
        """

        self.groups = groups


    def create_group_graphs(self):
        """
        Creates a dictionary of graphs for each group.

        Returns:
        - dict: Dictionary where keys are `group_id` and values are NetworkX Graphs.
        """

        group_graphs = {}

        for group in self.groups:
            G = nx.complete_graph(group.group_size)

            # Relabel nodes with person IDs
            mapping = {i: agent.agent_id for i, agent in enumerate(group.members)}
            G = nx.relabel_nodes(G, mapping)

            # Assign node attributes
            for agent in group.members:
                G.nodes[agent.agent_id].update(agent.__dict__)

            group_graphs[group.group_id] = G

        return group_graphs

    

