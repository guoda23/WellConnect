import pandas as pd

class DataHandler:
    def __init__(self, data_path):
        """Initialize the DataHandler"""
        self.data_path = data_path

    def read_data(self, file_path, file_type="csv"):
        """
        Reads data from a file using pandas.
        
        Parameters:
        - file_path (str): Path to the data file.
        - file_type (str): The type of the file (default is "csv").
        
        Returns:
        - pd.DataFrame: The loaded data as a pandas DataFrame.
        """

        if file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "excel":
            return pd.read_excel(file_path)
        elif file_type == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
