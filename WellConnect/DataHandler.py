import pandas as pd
from entities.Agent import Agent

class DataHandler:
    def __init__(self, data_path, file_type="csv"):
        """Initialize the DataHandler
        
        Parameters:
        - data_path (str): Path to the data file.
        - file_type (str): The type of the file (default is "csv").
        """
        self.data_path = data_path
        self.file_type=file_type


    def read_data(self):
        """
        Reads data from a file using pandas.
        
        Returns:
        - pd.DataFrame: The loaded data as a pandas DataFrame.
        """

        if self.file_type == "csv":
            return pd.read_csv(self.data_path)
        elif self.file_type == "excel":
            return pd.read_excel(self.data_path)
        elif self.file_type == "json":
            return pd.read_json(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")


    def create_agents(self, data_df, relevant_attributes, id_column=None): #TODO:impute missing values?
        #TODO: move this elsewhere
        """
        Creates a list of Agent objects from the dataset.

        Parameters:
        - data_df (pd.DataFrame): The dataset containing rows of data.
        - relevant_attributes (list[str]): List of attributes to include for each Agent.
        - id_column (str, optional): The name of the column to use as the Agent ID.
        If not specified or if the column is missing, the DataFrame index will be used.

        Returns:
        - list[Agent]: A list of Agent objects.
        """

        agents = []

        for index, row in data_df.iterrows():
            agent_id = row[id_column] if id_column and id_column in data_df.columns else index
            relevant_attribute_data = {key: row[key] for key in relevant_attributes if key in row}
            agent = Agent(agent_id=agent_id, attribute_dict=relevant_attribute_data)
            agents.append(agent)

        return agents
