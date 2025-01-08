import networkx as nx

class NetworkBuilder:
    def __init__(self, group_data, group_size):
        """
        Initializes the NetworkBuilder.

        Parameters:
        - group_data (pd.DataFrame): DataFrame with `group_id` and individual attributes.
        - group_size (int): Number of individuals per group.
        """

        self.group_data = group_data
        self.group_size = group_size

    def create_group_graphs(self):
        """
        Creates a dictionary of graphs for each group.

        Returns:
        - dict: Dictionary where keys are `group_id` and values are NetworkX Graphs.
        """

        group_graphs = {}

        for group_id, group_data in self.group_data.groupby('group_id'):
            person_ids = group_data.index.tolist()
            G = nx.complete_graph(self.group_size)

            # Relabel nodes with person IDs
            mapping = {i: person_ids[i] for i in range(self.group_size)}
            G = nx.relabel_nodes(G, mapping)

            # Assign node attributes
            for person_id, row in group_data.iterrows():
                G.nodes[person_id].update(row.to_dict())

            group_graphs[group_id] = G

        return group_graphs

    

