import pandas as pd


class PAdict:
    """
    Class to store the performance analysis of a model.
    It contains the AGN fraction, the count of images, and the evaluation metrics.
    """
    def __init__(self, model_name: str):
        """
        Initialize the PAdict class.

        :param model_name: Name of the model.
        :type model_name: str
        """
        self._model_name = model_name
        self._data = {}

    def __str__(self):
        """
        Return a string representation of the PAdict instance.
        
        :return: String representation of the PAdict instance.
        :rtype: str
        """
        return str(self.get_df())
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two PAdict instances are equal.

        :param other: Another object to compare with.
        :type other: object
        :return: True if the two PAdict indecies and columns are equal, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, PAdict):
            return False

        df = self.get_df()
        df_other = other.get_df()

        if (df.index.equals(df_other.index) and df.columns.equals(df_other.columns)):
            return True
        else:
            return False
        
    @property
    def data(self) -> dict:
        """
        Get the performance analysis data.
        The data is stored in a dictionary format.

        :return: Dictionary containing the performance analysis data.
        :rtype: dict
        """
        return self._data
    
    @property
    def model_name(self) -> str:
        """
        Get the model name.
        """
        return self._model_name
    
    def get_df(self, sort_by_fagn: bool = False) -> pd.DataFrame:
        """
        Get the performance analysis DataFrame.
        The rows are the AGN fractions and the columns are the performance metrics.

        :param sort_by_fagn: Whether to sort the DataFrame by AGN fraction.
        :type sort_by_fagn: bool
        :return: DataFrame containing the performance analysis.
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame({})

        for key, value in self._data.items(): # (f = ?*, {"count": ?, "evaluation": {}})
            count, evaluation = value["count"], value["evaluation"]
            
            for k, v in evaluation.items(): # (metric, value)
                df.loc[f"{key} ({count})", k] = v # row: `f_agn = ?* (?*)`; column: metric

        if sort_by_fagn:
            df.sort_index(inplace=True)

        return df

    def add(self, agn_fraction: str, count: int, evaluation: dict) -> None:
        """
        Add a new entry to the performance analysis dictionary.
        The entry contains the AGN fraction, the count of images, and the evaluation metrics.
        
        :param agn_fraction: AGN fraction.
        :type agn_fraction: str
        :param count: Count of images.
        :type count: int
        :param evaluation: Evaluation metrics.
        :type evaluation: dict
        """
        self._data[agn_fraction] = {
            "count": count,
            "evaluation": evaluation
        }

    def get_dict(self) -> dict:
        """
        Get the performance analysis dictionary.
        The dictionary contains the AGN fraction, the count of images, and the evaluation metrics.
        
        :return: Dictionary containing the performance analysis.
        :rtype: dict
        """
        return self._data
    
    @data.setter
    def data(self, data: dict) -> None:
        """
        Set the performance analysis data.
        The data is stored in a dictionary format.

        :param data: Dictionary containing the performance analysis data.
        :type data: dict
        """
        # Validate the input data
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        for key, value in data.items():
            if not isinstance(key, str):
                raise ValueError("Key must be a string.")
            if not isinstance(value, dict):
                raise ValueError("Value must be a dictionary.")
            if "count" not in value or "evaluation" not in value:
                raise ValueError("Value must contain 'count' and 'evaluation' keys.")
            if not isinstance(value["count"], int):
                raise ValueError("'count' must be an integer.")
            if not isinstance(value["evaluation"], dict):
                raise ValueError("'evaluation' must be a dictionary.")
            if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in value["evaluation"].items()):
                raise ValueError("All keys in 'evaluation' must be strings and values must be int or float.")
        self._data = data
    
    def summarize_performance(self):
        """
        Summarize the performance analysis by calculating the weighted
        average of each metric across the agn fractions.
        The summary is stored in a DataFrame.
        The rows are the performance metrics and the columns are the model name.

        Disclaimer: The `PSF PSNR` is calculated differently from the other metrics.
        f_agn = 0.0 is not included in the calculation as it's value is float('-inf')
        due to the PSF image being an array of 0s.
        """
        df = pd.DataFrame({})

        aux_dict = {} # Auxiliary dictionary to store the sum of the metrics
        counts_sum = 0
        psf_psnr_sum = 0 # PSF PSNR has a different treatment
        for key, value in self._data.items(): # (f = ?*, {"count": ?, "evaluation": {}})
            count, evaluation = value["count"], value["evaluation"]
            counts_sum += count

            if key != "f_agn = 0.0": # PSF PSNR is not calculated for f_agn = 0
                psf_psnr_sum += count # PSF PSNR has a different treatment

            for k, v in evaluation.items(): # (metric, value)
                if k == "PSF PSNR" and key == "f_agn = 0.0": # PSF PSNR is not calculated for f_agn = 0
                    continue

                if k not in aux_dict:
                    aux_dict[k] = 0

                aux_dict[k] += v * count # Weighted sum of the metric

        # Normalize each result in the auxiliary by the sum of the counts and add it to the master df
        for k, v in aux_dict.items():
            if k == "PSF PSNR": # PSF PSNR has a different treatment
                df.loc[k, self._model_name] = v / psf_psnr_sum
            else:
                df.loc[k, self._model_name] = v / counts_sum

        return df
    