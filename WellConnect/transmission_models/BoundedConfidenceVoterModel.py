import numpy as np

class BoundedConfidenceVoterModel:
    """Simulates depression transmission using a bounded confidence update."""

    def __init__(self, threshold=0.5, mu=0.5, steps=50,
                 brooding_weight=0.5, reflecting_weight=0.5):
        self.threshold = threshold
        self.mu = mu
        self.steps = steps
        self.brooding_weight = brooding_weight
        self.reflecting_weight = reflecting_weight
        self.history = []


    def update_state(self, G, scores):
        """Perform a single update step and return the new scores."""
        new_scores = scores.copy()
        
        for node in G.nodes:
            relevant_scores = []

            for neighbor in G[node]:
                weight = G[node][neighbor].get('weight')

                if weight is not None and weight >= self.threshold:
                    relevant_scores.append(scores[neighbor])

            if relevant_scores:
                neighbor_mean = np.mean(relevant_scores)
                crt_neg = G.nodes[node].get('crt_neg_total', 0)
                freq_neg = G.nodes[node].get('crt_freq_neg', 0)
                crt_pos = G.nodes[node].get('crt_pos_total', 0)
                freq_pos = G.nodes[node].get('crt_freq_pos', 0)

                brooding = self.brooding_weight * (crt_neg + freq_neg) / 10.0
                reflection = self.reflecting_weight * (crt_pos + freq_pos) / 10.0

                processing_bias = np.clip(brooding - reflection, -1, 1)
                mu_i = self.mu * processing_bias
                new_scores[node] = np.clip(
                    scores[node] + mu_i * (neighbor_mean - scores[node]),
                    0,
                    27,
                )

        return new_scores


    def run(self, obj):
        """Run the bounded confidence update on a graph or a Group."""
        self.history = []
        G = obj.network if hasattr(obj, "network") else obj
        nodes = list(G.nodes) if not hasattr(obj, "members") else obj.members

        scores = {n: G.nodes[n].get('phq9', 0) for n in nodes}

        for _ in range(self.steps):
            scores = self.update_state(G, scores)
            self.history.append([scores[n] for n in nodes])

        return np.array(self.history), nodes


    def get_history(self):
        """Return the history of scores after calling :meth:`run`."""
        return np.array(self.history)
