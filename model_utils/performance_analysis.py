import pandas as pd
import numpy as np


from utils import print_box


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
        self._fluxes_keys = ["real_psf_fluxes","predicted_psf_fluxes","real_gal_fluxes","predicted_gal_fluxes"]
        self._data = {} # key: f = ?*; value: {"count": ?, "evaluation": {}}

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
        # Remove the 'f_agn = 0.5' key if it exists
        if "f_agn = 0.5" in self._data:
            self._data.pop("f_agn = 0.5")

        df = pd.DataFrame({})

        diff_treatment_metrics = [
            "PSNR PSF",  # PSNR PSF is not calculated for f_agn = 0.0
            "FRF PSF",  # FRF PSF Loss is not calculated for f_agn = 0.0
        ]

        small_metrics = ["FRF", "FRF PSF", "Centroid Error"]

        for key, value in self._data.items(): # (f = ?*, {"count": ?, "evaluation": {}})
            count, evaluation = value["count"], value["evaluation"]

            # Remove the fluxes from the evaluation dictionary
            # This is done to avoid the fluxes being included in the summary
            # The fluxes are not metrics, but they are included in the evaluation dictionary
            evaluation.pop("real_psf_fluxes", None)
            evaluation.pop("predicted_psf_fluxes", None)
            evaluation.pop("real_gal_fluxes", None)
            evaluation.pop("predicted_gal_fluxes", None)
            
            for k, v in evaluation.items(): # (metric, value)
                if not isinstance(v, (int, float)):
                    if k in diff_treatment_metrics and key == "f_agn = 0.0":
                        # Some metrics are not calculated for f_agn = 0
                        continue

                    # Filter outliers for small metrics
                    v = np.array(v)
                    if k in small_metrics:
                        v = v[v < 5]
                        v = v[v > -5]

                    v = sum(v) / len(v) # Average the values if they are not int or float

                    if k == "FRF" or k == "FRF PSF":
                        v -= 1
                df.loc[f"{key} ({count})", k] = v # row: `f_agn = ?* (?*)`; column: metric

        columns_to_drop = ["RFC", "RFC PSF","Reconstruction Loss PSF"]
        # Drop some columns
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        if sort_by_fagn:
            df.sort_index(inplace=True)

        df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        df[df > 100] = float('nan') # Replace values greater than 100 with NaN
        df = df.round(3)
        df[df < -100] = float('nan') # Replace values less than 100 with NaN

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
        self._data = data

    def get_flux_data(self):
        """
        Get the flux data from the performance analysis dictionary.
        The flux data is stored in a dictionary format.

        :return: Dictionary containing the flux data.
        The structure is as follows:
            key: f = ?*;
            value: {
                "count": int,
                "real_psf_fluxes": np.ndarray,
                "predicted_psf_fluxes": np.ndarray,
                "real_gal_fluxes": np.ndarray,
                "predicted_gal_fluxes": np.ndarray
            }
        :rtype: dict
        """
        # Remove the 'f_agn = 0.5' key if it exists
        if "f_agn = 0.5" in self._data:
            self._data.pop("f_agn = 0.5")

        flux_data = {}
        for key, value in self._data.items():  # (f = ?*, {"count": ?, "evaluation": {}})
            count, evaluation = value["count"], value["evaluation"]
            flux_entry = {k: v for k, v in evaluation.items() if k in self._fluxes_keys}
            flux_entry["count"] = count
            flux_data[key] = flux_entry

        # Sort by f = ?*
        flux_data = dict(sorted(flux_data.items(), key=lambda x: float(x[0].split("=")[1].strip())))

        return flux_data  # key: f = ?*; value: {"count": ?, ...}

    def get_all_fluxes_np(self):
        flux_data = self.get_flux_data()

        real_psf_fluxes = np.array([])
        predicted_psf_fluxes = np.array([])
        real_gal_fluxes = np.array([])
        predicted_gal_fluxes = np.array([])

        for key, value in flux_data.items():
            real_psf_fluxes = np.concatenate((real_psf_fluxes, value["real_psf_fluxes"]), axis=0)
            predicted_psf_fluxes = np.concatenate((predicted_psf_fluxes, value["predicted_psf_fluxes"]), axis=0)
            real_gal_fluxes = np.concatenate((real_gal_fluxes, value["real_gal_fluxes"]), axis=0)
            predicted_gal_fluxes = np.concatenate((predicted_gal_fluxes, value["predicted_gal_fluxes"]), axis=0)

        info = "Retrieved fluxes:"
        info += f"\nReal PSF Fluxes: {real_psf_fluxes.shape}"
        info += f"\nPredicted PSF Fluxes: {predicted_psf_fluxes.shape}"
        info += f"\nReal Galaxy Fluxes: {real_gal_fluxes.shape}"
        info += f"\nPredicted Galaxy Fluxes: {predicted_gal_fluxes.shape}"
        print_box(info)

        return real_psf_fluxes, predicted_psf_fluxes, real_gal_fluxes, predicted_gal_fluxes
    
    def summarize_metrics_performance(self):
        """
        Summarize the performance analysis by calculating the weighted
        average of each metric across the agn fractions.
        The summary is stored in a DataFrame.
        The rows are the performance metrics and the columns are the model name.

        Disclaimer: Some metrics are calculated differently from the other metrics.
        f_agn = 0.0 is not included in the calculation as it's value is very large
        due to the PSF image being an array of 0s which messes with some of the metrics.
        This metrics are stored in the `diff_treatment_metrics` list.
        """
        # Remove the 'f_agn = 0.5' key if it exists
        if "f_agn = 0.5" in self._data:
            self._data.pop("f_agn = 0.5")

        small_metrics = ["FRF", "FRF PSF", "Centroid Error"]

        diff_treatment_metrics = [
            "PSNR PSF",  # PSNR PSF is not calculated for f_agn = 0.0
            "FRF PSF",  # FRF PSF Loss is not calculated for f_agn = 0.0
        ]

        df = pd.DataFrame({})

        aux_dict = {} # Auxiliary dictionary to store the sum of the metrics
        counts_sum = 0
        diff_sum = 0 # Some metrics that fail at f = 0.0 have a different treatment
        for key, value in self._data.items(): # (f = ?*, {"count": ?, "evaluation": {}})
            count, evaluation = value["count"], value["evaluation"]

            # Remove the fluxes from the evaluation dictionary
            # This is done to avoid the fluxes being included in the summary
            # The fluxes are not metrics, but they are included in the evaluation dictionary
            evaluation = {k: v for k, v in evaluation.items() if k not in self._fluxes_keys}

            # Add the count to the sum of counts
            counts_sum += count

            if key != "f_agn = 0.0": # Some metrics are not calculated for f_agn = 0
                diff_sum += count # Some metrics have a different treatment

            for k, v in evaluation.items(): # (metric, value)
                if k in diff_treatment_metrics and key == "f_agn = 0.0":
                    # Some metrics are not calculated for f_agn = 0
                    continue

                if k not in aux_dict:
                    aux_dict[k] = 0

                if not isinstance(v, (int, float)):
                    # Depricated: Used to remove the 1 outlier from the FRF metric
                    if key == "f_agn = 0.65" and k == "FRF":
                        v = np.array(v)
                        bad_batch_idx = np.argmax(v)
                        v = np.delete(v, bad_batch_idx)

                    # Filter outliers for small metrics
                    v = np.array(v)
                    if k in small_metrics:
                        v = v[v < 20]
                        v = v[v > -20]

                    v = sum(v) / len(v) # Average the values if they are not int or float

                aux_dict[k] += v * count # Weighted sum of the metric

        # Normalize each result in the auxiliary by the sum of the counts and add it to the master df
        for k, v in aux_dict.items():
            if k in diff_treatment_metrics: # Some metrics have a different treatment
                df.loc[k, self._model_name] = v / diff_sum
            else:
                df.loc[k, self._model_name] = v / counts_sum

        # Depriated:
        # Remove the Reconstruction Loss PSF from the summary
        rows_to_drop = ["RFC", "RFC PSF", "Reconstruction Loss PSF"]
        for row in rows_to_drop:
            if row in df.index:
                df.drop(index=row, inplace=True)

        return df
    