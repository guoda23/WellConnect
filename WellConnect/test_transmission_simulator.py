# temporary script to test the transmission simulator via a saved group
import pickle
import matplotlib.pyplot as plt
from TransmissionSimulator import TransmissionSimulator

# Load your saved experiment and pull out one group
def load_group_from_pickle(pkl_path, idx=0):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    groups = data["groups"]
    print(f"Loaded {len(groups)} groups; testing group #{idx}")
    return groups[idx]

def test_hmdh_model_on_group(group, steps=50):
    print("\n--- Testing HMDhModel ---")
    from TransmissionSimulator import TransmissionSimulator

    sim = TransmissionSimulator(
        model_type='HMDhModel',
        # customize params here if needed
    )
    history, agents = sim.run(group, steps=steps)

    # history should be shape (steps, 3): each row = [# healthy, # mild, # depressed]
    H, M, D = history[:, 0], history[:, 1], history[:, 2]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(H, label="Healthy", color="green")
    plt.plot(M, label="Mild", color="orange")
    plt.plot(D, label="Depressed", color="red")
    plt.xlabel("Time step")
    plt.ylabel("Number of individuals")
    plt.title("Population-level Depression States Over Time (HMDhModel)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_voter_model_on_group(group, steps=5):
    print("\n--- Testing Bounded Confidence Voter Model ---")
    sim = TransmissionSimulator(
        model_type='BoundedConfidenceVoterModel',
        target_attr="PHQ9_Total",
        threshold=0.8,
        mu=0.2,
        # brooding_weight=0.5,
        # reflecting_weight=0.5
    )
    history, agents = sim.run(group, steps=steps)

    for i, agent in enumerate(agents):
        label = f'Agent {agent.agent_id}' if hasattr(agent, 'agent_id') else f'Agent {i}'
        plt.plot(history[:, i], lw=1, label=label)

    plt.xlabel("Time step")
    plt.ylabel("PHQ-9 score")
    plt.yticks([0, 5, 10, 15, 20, 27],
               labels=["None", "Mild", "Moderate", "Mod. Severe", "Severe", "Max"])
    plt.title("PHQ-9 Score Over Time (Voter Model)")
    plt.legend(ncol=2, fontsize="small", loc='upper right')
    plt.tight_layout()
    plt.show()
    # plt.savefig("voter_model_phq9_plot.png", dpi=300)
    # print("Plot saved as voter_model_phq9_plot.png")

if __name__ == "__main__":
    PKL = "Experiment_data/batch_2025-07-21_14-57-39/experiment_run_1/experiment_2025-07-21_14-57-39.pkl"
    group = load_group_from_pickle(PKL, idx=0)

    # test_hmdh_model_on_group(group, steps=50)
    test_voter_model_on_group(group, steps=10)
