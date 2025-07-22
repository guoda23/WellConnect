import numpy as np


class BoundedConfidenceVoterModel:
    """Simulates depression transmission using a bounded confidence update."""

    def __init__(self, target_attr="PHQ9_Total", threshold=0.5, mu=0.5,
                 brooding_weight=0.5, reflecting_weight=0.5):
        """
        Initialize the model with parameters.

        Parameters
        ----------
        target_attr : str
            Name of the agent attribute to be updated (e.g., 'phq9_total').
        threshold : float
            Minimum edge weight for influence.
        mu : float
            Base influence strength (how much people are affected by others).
        brooding_weight : float
            Weight for brooding (negative effect).
        reflecting_weight : float
            Weight for reflection (positive effect).
        """
        self.target_attr = target_attr #attribute which is spread in the group
        self.corumination_neg_total_attr = 'PANCRS_TotalPositive'
        self.corumination_pos_total_attr = 'PANCRS_TotalNegative'
        self.corumination_pos_freq_attr = 'PANCRS_FrequencyPositive'
        self.corumination_neg_freq_attr = 'PANCRS_FrequencyNegative'
        self.threshold = threshold
        self.mu = mu
        self.brooding_weight = brooding_weight
        self.reflecting_weight = reflecting_weight
        self.history = []


    def update_state(self, group_network):
        """
        Update all agents once using neighbor influence.

        Parameters
        ----------
        group_network : networkx.Graph
            A graph where nodes are Agent objects and edges have 'weight' attributes.

        Returns
        -------
        None
            Agents are updated in-place.
        """

        for agent in group_network.nodes:
            neighbours = list(group_network.neighbors(agent))
            relevant_scores = []

            for neighbour in neighbours:
                weight = group_network.edges[agent, neighbour]['weight']

                if weight >= self.threshold:
                    score = getattr(neighbour, self.target_attr)
                    relevant_scores.append(score)

            #if there are neighbours of sufficient closeness update the state
            if relevant_scores: 
                neighbour_mean = np.mean(relevant_scores)

                crt_neg = getattr(agent, self.corumination_neg_total_attr)
                freq_neg = getattr(agent, self.corumination_pos_total_attr)
                crt_pos = getattr(agent, self.corumination_pos_freq_attr)
                freq_pos = getattr(agent, self.corumination_neg_freq_attr)

                brooding = self.brooding_weight * (crt_neg + freq_neg) / 10.0
                reflection = self.reflecting_weight * (crt_pos + freq_pos) / 10.0
                processing_bias = np.clip(brooding - reflection) # min -1, max 1
                
                current = getattr(agent, self.target_attr)
                mu_i = self.mu * processing_bias
                updated = np.clip(current + mu_i * (neighbour_mean - current), 0, 27)

                setattr(agent, self.target_attr, updated)


    def run(self, group, steps=50):
        """
        Run the model on a Group for a number of steps.

        Parameters
        ----------
        group : Group
            A Group object with a .network attribute (NetworkX graph of Agent nodes).
        steps : int
            Number of update steps to run.

        Returns
        -------
        history : np.ndarray
            Array of scores over time (steps × agents).
        agents : list
            List of Agent objects in the same order as the history columns.
        """
        self.history = []
        group_network = group.network
        agents = list(group_network.nodes)

        for _ in range(steps):
            self.update_state(group_network)
            step_scores = [getattr(agent, self.target_attr) for agent in agents]
            self.history.append(step_scores)

        return np.array(self.history), agents


    def get_history(self):
        """
        Get the recorded simulation history.

        Returns
        -------
        history : np.ndarray
            History of scores from the most recent run.
        """
        return np.array(self.history)
