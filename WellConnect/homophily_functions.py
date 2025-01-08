def linear_homophily_function():
    pass

#Functoin registry
HOMOPHILY_FUNCTIONS = {
    "linear": linear_homophily_function,
}


def linear_homophily_function(node1, node2, G, weights, max_distances): #TODO: incorporate euclidean distance?
    """
    Calculates the homophily score (e.g., tie strength between two individuals based on their trait compatibility).

    Parameters:
        node1 (int): The first node in the graph (representing inidividual 1).
        node2 (int): The second node in the graph (representing individual 2).
        G (networkx.Graph): The graph containing the nodes.
        weights (dict): Weights for each attribute.
        max_distances (dict): Maximum possible distances for each trait (for normalization purposes).

    Returns:
        float: Homophily score between 0 and 1.
    """

    normalized_total_distance = 0
    max_normalized_total_distance = sum(weights.values())

    for attribute, weight in weights.items():
        value1 = G.nodes[node1].get(attribute)
        value2 = G.nodes[node2].get(attribute)
        max_attribute_distance = max_distances[attribute]

        #handle continuous variables
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)): #TODO: add handling of NA values
            absolute_distance = abs(value1 - value2)
            normalized_distance = absolute_distance / max_attribute_distance
            weighted_normalized_distance = normalized_distance * weight
            normalized_total_distance += weighted_normalized_distance

        #handle categorical variables
        elif isinstance(value1, str) and isinstance(value2, str):
            normalized_distance = 1 if value1 != value2 else 0
            normalized_total_distance += weight * normalized_distance

    #Convert distance to hours
    #TODO: use a nonmonotonic function here according to Bruggeman
    # Output score between 0 and 1
    score = 1 - (normalized_total_distance / max_normalized_total_distance)
    return round(score, 2)
    pass