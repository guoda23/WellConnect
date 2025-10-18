#input should be one group + params relating to the spread
#output should be a data structure returning the spread log

class TransmissionSimulator:
    """
    Wrapper for running a selected transmission model.
    """
    def __init__(self, model_type, **kwargs):
        """
        Wrapper for running a selected transmission model.
        """
        self.model = self._initialize_model(model_type, **kwargs)

    def _initialize_model(self, model_type, **kwargs):
        """
        Initialize the simulator with a specific model type and appropriate parameters.

        Parameters
        ----------
        model_type : str
            The type of model to use. Options: 'epidemic', 'voter'.
        **kwargs :
            Model-specific keyword arguments passed to the chosen model class.
        """
        if model_type == "HMDaModel":
            from transmission_models.HMDaModel import HMDaModel
            return HMDaModel(**kwargs)
        elif model_type == "BoundedConfidenceVoterModel":
            from transmission_models.BoundedConfidenceVoterModel import BoundedConfidenceVoterModel
            return BoundedConfidenceVoterModel(**kwargs)
        elif model_type == "AlcoholHMDaModel":
            from transmission_models.AlcoholHMDaModel import AlcoholHMDaModel
            return AlcoholHMDaModel(**kwargs)
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized.")
        
    def run(self, group, steps):
        """
        Runs the selected model and returns its output.
        """
        return self.model.run(group, steps)