from DataHandler import DataHandler
from entities.Agent import Agent

# Define the input data path and attributes
data_path = "binary_age_gender_edu.csv"  # Replace with the path to your CSV file
file_type = "csv"

# Relevant attributes to use for creating agents
relevant_attributes = ["age", "gender", "education"]

# Initialize the DataHandler
data_handler = DataHandler(data_path=data_path, file_type=file_type)

# Step 1: Read the data
print("Reading data from file...")
data_df = data_handler.read_data()
print("Data loaded:")
print(data_df.head())

# Step 2: Create agents from the data
print("\nCreating agents...")
agents = data_handler.create_agents(data_df, relevant_attributes, id_column='id')

# Step 3: Inspect the created agents
print(f"\nCreated {len(agents)} agents. Showing details of the first 5 agents:")
for agent in agents[:5]:  # Show the first 5 agents for brevity
    print(agent)

# Step 4: Verify attributes of a specific agent
print("\nVerifying attributes of the first agent:")
if agents:
    first_agent = agents[0]
    print("Agent's __dict__:", first_agent.__dict__)
    print("Has 'age' attribute:", hasattr(first_agent, "age"))
    print("Has 'income' attribute:", hasattr(first_agent, "income"))
