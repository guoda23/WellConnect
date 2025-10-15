import numpy as np
import networkx as nx

class HMDaModel:
    """
    HMDaModel — Hill-style automatic contagion model for depression
    (adapted from Hill et al., 2010, *Proc. R. Soc. B*).

    This version uses weekly time steps (Hill's per-year rates divided by 52)
    and the PHQ-9 categories used in the UMH study.

    States:
        0 = Healthy          (content analogue)
        1 = Mildly Depressed (neutral analogue)
        2 = Depressed        (discontent analogue)

    Transition structure (from Hill et al., Fig. 3):
        Mild → Healthy:      a_h + b_h·n_H
        Mild → Depressed:    a_d + b_d·n_D
        Healthy → Mild:      g_h
        Depressed → Mild:    g_d
        Healthy → Depressed: s_hd  (superinfection)
        Depressed → Healthy: s_dh  (superinfection)

    Default simulation duration: 20 weeks
        - 8 weekly meetings
        - 4 biweekly meetings (8 additional weeks)
    """

    def __init__(self,
                 seed,
                 # Hill’s yearly rates divided by 52 (weekly timescale)
                 a_h=0.18/52,  b_h=0.02/52,  g_h=0.088/52,
                 a_d=0.04/52,  b_d=0.04/52,  g_d=0.13/52,
                 s_hd=0.009/52, s_dh=0.07/52,
                 state_attr="depression_state",
                 phq9_attr="PHQ9_Total"):

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.state_attr = state_attr
        self.phq9_attr = phq9_attr
        self.history = []

        self.p = dict(a_h=a_h, b_h=b_h, g_h=g_h,
                      a_d=a_d, b_d=b_d, g_d=g_d,
                      s_hd=s_hd, s_dh=s_dh)

    # ------------------------------------------------------------------
    def classify_depression_state(self, phq_score):
        """Classify PHQ-9 score into Healthy (0), Mild (1), or Depressed (2)."""
        if phq_score < 5:
            return 0
        elif phq_score < 10:
            return 1
        else:
            return 2

    # ------------------------------------------------------------------
    def initialize_agent_states(self, group):
        """Assign initial depression state from PHQ-9 scores."""
        self.g = group.network
        for agent in self.g.nodes:
            score = getattr(agent, self.phq9_attr)
            setattr(agent, self.state_attr, self.classify_depression_state(score))

    # ------------------------------------------------------------------
    def update_state(self):
        """Perform one weekly update according to Hill-style transition rules."""
        agents = list(self.g.nodes)
        state = np.array([getattr(a, self.state_attr) for a in agents])
        A = nx.to_numpy_array(self.g, nodelist=agents, weight="weight")

        p = self.p

        # Weighted counts of healthy and depressed neighbours
        nH = A @ (state == 0)
        nD = A @ (state == 2)

        # Weekly transition probabilities
        p_mh = (p["a_h"] + p["b_h"] * nH)           # Mild → Healthy
        p_md = (p["a_d"] + p["b_d"] * nD)           # Mild → Depressed
        p_hm = np.full_like(p_mh, p["g_h"])         # Healthy → Mild
        p_dm = np.full_like(p_mh, p["g_d"])         # Depressed → Mild
        p_hd = np.full_like(p_mh, p["s_hd"])        # Healthy → Depressed (superinfection)
        p_dh = np.full_like(p_mh, p["s_dh"])        # Depressed → Healthy (superinfection)

        # Random draws for stochastic transitions
        draws = self.rng.random((len(agents), 3))
        new_state = state.copy()

        # Mild transitions
        maskM = state == 1
        new_state[maskM & (draws[:,0] < p_mh)] = 0
        new_state[maskM & (draws[:,1] < p_md)] = 2

        # Healthy transitions
        maskH = state == 0
        new_state[maskH & (draws[:,0] < p_hm)] = 1
        new_state[maskH & (draws[:,1] < p_hd)] = 2

        # Depressed transitions
        maskD = state == 2
        new_state[maskD & (draws[:,0] < p_dm)] = 1
        new_state[maskD & (draws[:,1] < p_dh)] = 0

        # Commit updated states to agents
        for i, agent in enumerate(agents):
            setattr(agent, self.state_attr, int(new_state[i]))

    # ------------------------------------------------------------------
    def run(self, group, steps=20):
        """
        Run the simulation for a number of weekly steps (default 20 weeks).

        Returns:
            history (np.ndarray): time-series of counts [healthy, mild, depressed]
            agents (list): final agent objects
        """
        self.g = group.network
        self.initialize_agent_states(group)
        agents = list(self.g.nodes)
        self.history = []

        for _ in range(steps):
            counts = [0, 0, 0]
            for a in agents:
                counts[getattr(a, self.state_attr)] += 1
            self.history.append(counts)
            self.update_state()

        return np.array(self.history), agents
