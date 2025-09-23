from homophily_functions import HOMOPHILY_FUNCTIONS

class ConnectionPredictor:
    def __init__(self, weights, max_distances, homophily_function_name = 'linear_deterministic'):
        """
        Initializes the ConnectionPredictor.

        Parameters:
        - homophily_function_name (str, optional): Name of the homophily function to use.
          Defaults to "linear_deterministic".
        - weights (dict): Weights for attributes.
        - max_distances (dict): Maximum distances within attributes for normalization.
        """
        if homophily_function_name not in HOMOPHILY_FUNCTIONS:
            raise ValueError(f"Homophily function '{homophily_function_name}' not found. "
                             f"Available functions: {list(HOMOPHILY_FUNCTIONS.keys())}")
        
        self.homophily_function = HOMOPHILY_FUNCTIONS[homophily_function_name]
        self.weights = weights
        self.max_distances = max_distances
        

    def predict_weights(self, G):
        """
        Assigns predicted weights to edges in group graphs using the homophily function.

        Parameters:
        - G (networkx.Graph): The graph to predict connection weights for.

        Returns:
        - networkx.Graph: Updated graph with edge weights assigned.
        """ 
        
        for agent1, agent2 in G.edges():
            weight = self.homophily_function(
                agent1=agent1,
                agent2=agent2,
                weights=self.weights,
                max_distances=self.max_distances
            )
            G.edges[agent1, agent2]['weight'] = weight
        return G
    