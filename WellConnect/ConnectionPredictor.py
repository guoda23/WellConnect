from homophily_functions import HOMOPHILY_FUNCTIONS

class ConnectionPredictor:
    def __init__(self, weights, max_distances, homophily_function_name = 'linear'):
        """
        Initializes the ConnectionPredictor.

        Parameters:
        - homophily_function_name (str, optional): Name of the homophily function to use.
          Defaults to "linear".
        - weights (dict): Weights for attributes.
        - max_distances (dict): Maximum distances within attributes for normalization.
        """
        if homophily_function_name not in HOMOPHILY_FUNCTIONS:
            raise ValueError(f"Homophily function '{homophily_function_name}' not found. "
                             f"Available functions: {list(HOMOPHILY_FUNCTIONS.keys())}")
        
        self.homophily_function = HOMOPHILY_FUNCTIONS[homophily_function_name]
        self.weights = weights
        self.max_distances = max_distances
        

    def predict_weights(self, group_graph_dict):
        """
        Assigns predicted weights to edges in group graphs using the homophily function.

        Parameters:
            group_graph_dict (dict): Dictionary of graphs, keyed by group ID.

        Returns:
            dict: Updated group graphs with edge weights assigned.
        """ 

        for group_id, G in group_graph_dict.items():
            for node1, node2 in G.edges():
                weight = self.homophily_function(node1, node2, G, self.weights, self.max_distances)
                G.edges[node1, node2]['weight'] = weight
        return group_graph_dict
    