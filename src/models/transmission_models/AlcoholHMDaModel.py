import numpy as np
import networkx as nx

# Note: This model was adapted from van der Ende et al. (2024). This is the Github link to their original code:
# https://github.com/MwjEnde/AMHa_code/blob/main/AMHa_code/simulations/AMHA_class.py

class AlcoholHMDaModel:
    """
    Simulates the spread and recovery of depressive symptoms in a social network of agents.

    Each agent is a node in a weighted networkx graph (`group.network`) and must have a PHQ-9 score 
    (attribute name defined by `phq9_attr`). At initialization, the model classifies agents into 
    three states based on their PHQ-9 score:
        0 = Mild
        1 = Moderate
        2 = Severe

    The model uses empirically derived transition parameters from Van der Ende et al. (2024), 
    calibrated on an unweighted network, to simulate:
        - Spontaneous transitions (α): baseline probability of moving to another state
        - Social influence (β): increased transition likelihood based on neighbor states

    Constants are scaled using global multipliers (e.g., alpha_i_mp, beta_r_mp), 
    and optional fine-tuned individual multipliers (`mtp_*_individual`) per transition.
    """

    def __init__(self,
                 alpha_i_mp=0, # Set to 0 to isolate social influences!
                 alpha_r_mp=0, # Set to 0 to isolate social influences!
                 beta_i_mp=1,
                 beta_r_mp=1,
                 mtp_a_i_individual=np.array([1, 1, 1]),
                 mtp_a_r_individual=np.array([1, 1, 1]),
                 mtp_b_i_individual=np.array([1, 1, 1, 1]),
                 mtp_b_r_individual=np.array([1, 1, 1]),
                 flat_ai=0, flat_ar=0, flat_bi=0, flat_br=0,
                 state_attr="depression_state",
                 phq9_attr="PHQ9_Total"):
        
        self.state_attr = state_attr
        self.phq9_attr = phq9_attr
        self.history = []

        self.alpha_i_mp = alpha_i_mp 
        self.alpha_r_mp = alpha_r_mp 
        self.beta_i_mp = beta_i_mp
        self.beta_r_mp = beta_r_mp

        self.mtp_a_i_individual = mtp_a_i_individual 
        self.mtp_a_r_individual = mtp_a_r_individual
        self.mtp_b_i_individual = mtp_b_i_individual
        self.mtp_b_r_individual = mtp_b_r_individual

        self.flat_ai = flat_ai
        self.flat_ar = flat_ar
        self.flat_bi = flat_bi
        self.flat_br = flat_br

        mtp_ai = self.alpha_i_mp * self.mtp_a_i_individual
        mtp_ar = self.alpha_r_mp * self.mtp_a_r_individual
        mtp_bi = self.beta_i_mp * self.mtp_b_i_individual
        mtp_br = self.beta_r_mp * self.mtp_b_r_individual

        self.constants = { 
            'mi_mo': np.array([0.2108 * mtp_ai[0] + flat_ai, 0.0202 * mtp_bi[0] + flat_bi]) / 239.2,
            'mi_s': np.array([0.0081 * mtp_ai[1] + flat_ai, 0.0000 * mtp_bi[1] + flat_bi]) / 239.2,
            'mo_mi': np.array([0.1942 * mtp_ar[0] + flat_ar, 0.0348 * mtp_br[0] + flat_br]) / 239.2,
            'mo_s': np.array([0.0839 * mtp_ai[2] + flat_ai, 0.0357 * mtp_bi[2] + flat_bi]) / 239.2,
            's_mi': np.array([0.0558 * mtp_ar[1] + flat_ar, 0.0190 * mtp_br[1] + flat_br]) / 239.2,
            's_mo': np.array([0.2997 * mtp_ar[2] + flat_ar, 0.0000 * mtp_br[2] + flat_br]) / 239.2,
            'mi_mo_s': np.array([0.0271 * mtp_bi[3] + flat_bi]) / 239.2,
        }

        self.transition_log = []

        self.seed = None
        self.rng = None


    def log_transitions(self, old_states, new_states):
        state_names = {0: "Mi", 1: "Mo", 2: "S"}
        step_transitions = {}

        for old, new in zip(old_states, new_states):
            if old != new:
                label = f"{state_names[old]}→{state_names[new]}"
                step_transitions[label] = step_transitions.get(label, 0) + 1

        self.transition_log.append(step_transitions)


    def classify_depression_state(self, phq_score):
        if phq_score < 10:
            return 0  # Mild
        elif phq_score < 15:
            return 1  # Moderate
        else:
            return 2  # Severe


    def initialize_agent_states(self, group):
        self.g = group.network
        for agent in self.g.nodes:
            score = getattr(agent, self.phq9_attr)
            setattr(agent, self.state_attr, self.classify_depression_state(score))


    def update_state(self):
        constants = self.constants
        agents = list(self.g.nodes)

        state = np.array([getattr(agent, self.state_attr) for agent in agents])
        old_state = state.copy()

        mild_idx = np.where(state == 0)[0]
        moderate_idx = np.where(state == 1)[0]
        severe_idx = np.where(state == 2)[0]

        vec_mild = (state == 0).astype(float)
        vec_moderate = (state == 1).astype(float)
        vec_severe = (state == 2).astype(float)

        A = nx.to_numpy_array(self.g, nodelist=agents, weight="weight")

        num_mi_mo = A @ vec_moderate
        num_mi_s = A @ vec_severe
        num_mo_mi = A @ vec_mild
        num_mo_s = A @ vec_severe
        num_s_mi = A @ vec_mild
        num_s_mo = A @ vec_moderate

        mi_to_mo_prob = constants['mi_mo'][0] + num_mi_mo * constants['mi_mo'][1] + num_mi_s * constants['mi_mo_s']
        mi_to_s_prob = constants['mi_s'][0] + num_mi_s * constants['mi_s'][1]
        mo_to_mi_prob = constants['mo_mi'][0] + num_mo_mi * constants['mo_mi'][1]
        mo_to_s_prob = constants['mo_s'][0] + num_mo_s * constants['mo_s'][1]
        s_to_mi_prob = constants['s_mi'][0] + num_s_mi * constants['s_mi'][1]
        s_to_mo_prob = constants['s_mo'][0] + num_s_mo * constants['s_mo'][1]

        draw_mi_mo = self.rng.random(len(mild_idx))
        draw_mi_s = self.rng.random(len(mild_idx))
        draw_mo_mi = self.rng.random(len(moderate_idx))
        draw_mo_s = self.rng.random(len(moderate_idx))
        draw_s_mi = self.rng.random(len(severe_idx))
        draw_s_mo = self.rng.random(len(severe_idx))

        state[mild_idx[mi_to_mo_prob[mild_idx] > draw_mi_mo]] = 1
        state[mild_idx[mi_to_s_prob[mild_idx] > draw_mi_s]] = 2
        state[moderate_idx[mo_to_mi_prob[moderate_idx] > draw_mo_mi]] = 0
        state[moderate_idx[mo_to_s_prob[moderate_idx] > draw_mo_s]] = 2
        state[severe_idx[s_to_mi_prob[severe_idx] > draw_s_mi]] = 0
        state[severe_idx[s_to_mo_prob[severe_idx] > draw_s_mo]] = 1

        for i, agent in enumerate(agents):
            setattr(agent, self.state_attr, int(state[i]))

        self.log_transitions(old_state, state)


    def _count_states(self, agents):
        counts = [0, 0, 0]
        for agent in agents:
            counts[getattr(agent, self.state_attr)] += 1
        return counts


    def run(self, group, seed, steps=20):
        self.history = []
        self.transition_log = []
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.g = group.network
        self.initialize_agent_states(group)
        agents = list(self.g.nodes)

        self.history = [self._count_states(agents)]

        for _ in range(steps):
            self.update_state()
            self.history.append(self._count_states(agents))

        return np.array(self.history), agents, self.transition_log
