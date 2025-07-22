import numpy as np
import networkx as nx

# Note: This model was adapted from van der Ende et al. (2024). This is the Github link to their original code:
# https://github.com/MwjEnde/AMHa_code/blob/main/AMHa_code/simulations/AMHA_class.py

class HMDhModel:
    """
    Simulates the spread and recovery of depressive symptoms in a social network of agents.

    Each agent is a node in a weighted networkx graph (`group.network`) and must have a PHQ-9 score 
    (attribute name defined by `phq9_attr`). At initialization, the model classifies agents into 
    three states based on their PHQ-9 score:
        0 = Healthy
        1 = Mildly Depressed
        2 = Depressed

    The model uses empirically derived transition parameters from Van der Ende et al. (2024), 
    calibrated on an unweighted network, to simulate:
        - Spontaneous transitions (α): baseline probability of moving to another state
        - Social influence (β): increased transition likelihood based on neighbor states

    Constants are scaled using global multipliers (e.g., alpha_i_mp, beta_r_mp), 
    and optional fine-tuned individual multipliers (`mtp_*_individual`) per transition.
    """

    def __init__(self,
                 alpha_i_mp=1, #global multiplier for spontaneous transitions (alpha) in worsening directions (H->M, H->D, M->D)
                 alpha_r_mp=1, #global multiplier for spontaneous transitions (alpha) towards recovery states (M->H, D->H, D->M)
                 beta_i_mp=1, #global multiplier for social contagion into worse states
                 beta_r_mp=1, #global multiplier for social contagion into recovery states
                 mtp_a_i_individual=np.array([1, 1, 1]), #scaling global multiplier for spontaneous transitions in worsening directions (H->M, H->D, M->D)
                 mtp_a_r_individual=np.array([1, 1, 1]), #scaling global multiplier for spontaneous transitions towards recovery states (M->H, D->H, D->M)
                 mtp_b_i_individual=np.array([1, 1, 1, 1]), #scaling global multiplier for social contagion into worse states (H->M, H->D, M->D, D->M)
                 mtp_b_r_individual=np.array([1, 1, 1]), #scaling global multiplier for social contagion into recovery states (M->H, D->H, D->M)
                 flat_ai=0, flat_ar=0, flat_bi=0, flat_br=0, #control over base probabilities (intercept)
                 state_attr="depression_state",
                 phq9_attr="PHQ9_Total"):
        
        """
        Initialize model parameters and transition constants.

        Parameters are grouped into:
        - alpha_*: spontaneous transition rates
        - beta_*: social influence multipliers
        - mtp_*_individual: per-transition multipliers for more fine-grained control
        - flat_*: optional fixed offsets to baseline probabilities
        """
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

        self.constants = { #constants refer to Van der Ende's figure 3 in the paper (they only included significant ones)
            'h_m': np.array([0.2108 * mtp_ai[0] + flat_ai, 0.0202 * mtp_bi[0] + flat_bi]) / 46,
            'h_d': np.array([0.0081 * mtp_ai[1] + flat_ai, 0.0000 * mtp_bi[1] + flat_bi]) / 46,
            'm_h': np.array([0.1942 * mtp_ar[0] + flat_ar, 0.0348 * mtp_br[0] + flat_br]) / 46,
            'm_d': np.array([0.0839 * mtp_ai[2] + flat_ai, 0.0357 * mtp_bi[2] + flat_bi]) / 46,
            'd_h': np.array([0.0558 * mtp_ar[1] + flat_ar, 0.0190 * mtp_br[1] + flat_br]) / 46,
            'd_m': np.array([0.2997 * mtp_ar[2] + flat_ar, 0.0000 * mtp_br[2] + flat_br]) / 46,
            'h_m_d': np.array([0.0271 * mtp_bi[3] + flat_bi]) / 46,
        }

    def classify_depression_state(self, phq_score):
        """
        Categorizes an agent into 0 (Healthy), 1 (Mild), or 2 (Depressed) based on PHQ-9 score.
        """
        if phq_score < 5:
            return 0  # Healthy
        elif phq_score < 10:
            return 1  # Mild
        else:
            return 2  # Depressed

    def initialize_agent_states(self):
        """
        Sets each agent’s depression state based on their PHQ-9 score.
        """
        for agent in self.g.nodes:
            score = getattr(agent, self.phq9_attr)
            setattr(agent, self.state_attr, self.classify_depression_state(score))

    def update_state(self):
        """
        Performs a single simulation step:
        - Computes how many neighbors each agent has in each state (weighted)
        - Calculates transition probabilities using α and β values
        - Applies stochastic updates based on those probabilities
        """
        constants = self.constants
        agents = list(self.g.nodes)
        state = np.array([getattr(agent, self.state_attr) for agent in agents])

        healthy_idx = np.where(state == 0)[0]
        mild_idx = np.where(state == 1)[0]
        depressed_idx = np.where(state == 2)[0]

        vec_healthy = (state == 0).astype(float)
        vec_mild = (state == 1).astype(float)
        vec_depressed = (state == 2).astype(float)

        A = nx.to_numpy_array(self.g, nodelist=agents, weight="weight")

        num_h_m = A @ vec_mild
        num_h_d = A @ vec_depressed
        num_m_h = A @ vec_healthy
        num_m_d = A @ vec_depressed
        num_d_h = A @ vec_healthy
        num_d_m = A @ vec_mild

        h_to_m_prob = constants['h_m'][0] + num_h_m * constants['h_m'][1] + num_h_d * constants['h_m_d']
        h_to_d_prob = constants['h_d'][0] + num_h_d * constants['h_d'][1]
        m_to_h_prob = constants['m_h'][0] + num_m_h * constants['m_h'][1]
        m_to_d_prob = constants['m_d'][0] + num_m_d * constants['m_d'][1]
        d_to_h_prob = constants['d_h'][0] + num_d_h * constants['d_h'][1]
        d_to_m_prob = constants['d_m'][0] + num_d_m * constants['d_m'][1]

        draw_h_m = np.random.random_sample(len(healthy_idx))
        draw_h_d = np.random.random_sample(len(healthy_idx))
        draw_m_h = np.random.random_sample(len(mild_idx))
        draw_m_d = np.random.random_sample(len(mild_idx))
        draw_d_h = np.random.random_sample(len(depressed_idx))
        draw_d_m = np.random.random_sample(len(depressed_idx))

        state[healthy_idx[h_to_m_prob[healthy_idx] > draw_h_m]] = 1
        state[healthy_idx[h_to_d_prob[healthy_idx] > draw_h_d]] = 2
        state[mild_idx[m_to_h_prob[mild_idx] > draw_m_h]] = 0
        state[mild_idx[m_to_d_prob[mild_idx] > draw_m_d]] = 2
        state[depressed_idx[d_to_h_prob[depressed_idx] > draw_d_h]] = 0
        state[depressed_idx[d_to_m_prob[depressed_idx] > draw_d_m]] = 1

        for i, agent in enumerate(agents):
            setattr(agent, self.state_attr, int(state[i]))

    def run(self, group, steps=None):
        """
        Runs the model for a number of steps on a Group object (which must have `.network` attribute).

        Parameters:
        - group: Group object containing a networkx graph as `group.network`
        - steps: Number of simulation steps (defaults to `self.number_of_iterations` if defined)

        Returns:
        - A history array of counts in each state at each time step
        - The list of final agent objects
        """
        if steps is None:
            steps = self.number_of_iterations

        self.g = group.network
        self.initialize_agent_states()
        agents = list(self.g.nodes)
        self.history = []

        for _ in range(steps):
            counts = [0, 0, 0]
            for agent in agents:
                s = getattr(agent, self.state_attr)
                counts[s] += 1
            self.history.append(counts)
            self.update_state()

        return np.array(self.history), agents
