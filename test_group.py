from entities.Agent import Agent
from entities.Group import Group

def test_group_class():
    # Create test agents
    agent1 = Agent(agent_id=1, attribute_dict={"age": 25, "gender": "M", "education": "Bachelors"})
    agent2 = Agent(agent_id=2, attribute_dict={"age": 30, "gender": "F", "education": "Masters"})
    agent3 = Agent(agent_id=3, attribute_dict={"age": 22, "gender": "M", "education": "PhD"})

    print("=== Step 1: Create an empty group ===")
    # Create a group with no members initially
    group = Group(group_id=1)
    print("Initial Group:", group)

    print("\n=== Step 2: Add members to the group ===")
    # Add members to the group
    group.add_member(agent1)
    print("Group after adding agent1:", group)
    group.add_member(agent2)
    print("Group after adding agent2:", group)

    print("\n=== Step 3: Test group size constraint ===")
    # Test group size constraint
    try:
        group.add_member(agent3)
    except ValueError as e:
        print("Error:", e)

    print("\n=== Step 4: Retrieve member IDs and specific attributes ===")
    # Retrieve member IDs
    member_ids = group.get_member_ids()
    print("Member IDs:", member_ids)

    # Retrieve member attributes (e.g., ages)
    member_ages = group.get_member_attributes("age")
    print("Member Ages:", member_ages)

if __name__ == "__main__":
    test_group_class()
