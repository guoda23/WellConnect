#temporary script to test the transmission simulator
from TransmissionSimulator import TransmissionSimulator
from transmission_models.BoundedConfidenceVoterModel import BoundedConfidenceVoterModel
from transmission_models.HMDhModel import HMDhModel

def test_hmdh_model():
    print("\n--- Testing HMDhModel ---")
    simulator = TransmissionSimulator(
        model_type='HMDhModel',
        population_size=100,
        infection_rate=0.1,
        recovery_rate=0.05,
        initial_infected=5
    )
    results = simulator.run(steps=5)
    print("Results:", results)

def test_bounded_confidence_voter_model():
    print("\n--- Testing Bounded Confidence Voter Model ---")
    simulator = TransmissionSimulator(
        model_type='BoundedConfidenceVoterModel',
        population_size=100,
        initial_infected=5
    )
    results = simulator.run(steps=5)
    print("Results:", results)

if __name__ == "__main__":
    test_hmdh_model()
    test_bounded_confidence_voter_model()
