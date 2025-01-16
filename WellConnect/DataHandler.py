import pandas as pd

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
