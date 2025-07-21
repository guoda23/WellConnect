# temporary script to test the transmission simulator via a saved group
import pickle
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from TransmissionSimulator import TransmissionSimulator

# Load your saved experiment and pull out one group
def load_group_from_pickle(pkl_path, idx=0):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    groups = data["groups"]
    print(f"Loaded {len(groups)} groups; testing group #{idx}")
    return groups[idx]

def test_hmdh_model_on_group(group, steps=5):
    print("\n--- Testing HMDhModel ---")
    sim = TransmissionSimulator(
        model_type='HMDhModel',
        # replace toy params with real ones, if needed:
        infection_rate=0.1,
        recovery_rate=0.05,
    )
    history, agents = sim.run(group, steps=steps)
    print("First step:", history[0])
    print("Last  step:", history[-1])

def test_voter_model_on_group(group, steps=5):
    print("\n--- Testing Bounded Confidence Voter Model ---")
    sim = TransmissionSimulator(
        model_type='BoundedConfidenceVoterModel',
        target_attr="PHQ9_Total",
        threshold=0.5,
        mu=0.5,
        brooding_weight=0.5,
        reflecting_weight=0.5
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
    plt.savefig("voter_model_phq9_plot.png", dpi=300)
    print("Plot saved as voter_model_phq9_plot.png")

if __name__ == "__main__":
    PKL = "Experiment_data/batch_2025-07-21_14-57-39/experiment_run_1/experiment_2025-07-21_14-57-39.pkl"
    group = load_group_from_pickle(PKL, idx=0)

    # for agent in group.members:
    #     print(f"Agent {agent.agent_id}:")
    #     for attr, val in agent.__dict__.items():
    #         print(f"  {attr}: {val}")
    #     print()

    # test_hmdh_model_on_group(group, steps=50)
    test_voter_model_on_group(group, steps=50)
