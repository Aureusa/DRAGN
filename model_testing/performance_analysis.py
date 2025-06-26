import pandas as pd
import numpy as np


from utils import print_box
from utils_utils.warnings import AttributeMisuseWarning


class PAdict:
    """
    The PAdict class is used to store the performance analysis data
    in a dictionary format. It contains usefull infomation about the model's performance
    for different AGN fractions. The data is stored in a dictionary format where the keys are
    the AGN fractions (e.g., "f_agn = 0.0", "f_agn = 0.10", etc.) and the values are dictionaries
    containing the count of images in that specific bin of AGN fraction along with
    another dictionary containing per sample metric scores, along with the fluxes
    for the real and predicted PSF and galaxy images.
    """
    def __init__(self, model_name: str):
        """
        Initialize the PAdict class.

        :param model_name: Name of the model.
        :type model_name: str
        """
        self._model_name = model_name
        self._fluxes_keys = [
            "real_psf_fluxes",
            "predicted_psf_fluxes",
            "real_gal_fluxes",
            "predicted_gal_fluxes"
        ]
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
        
    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """
        Remove outliers from the given values using the IQR method.

        :param values: Array of values to remove outliers from.
        :type values: np.ndarray
        :return: Array of values with outliers removed.
        :rtype: np.ndarray
        """
        if len(values) == 0:
            return values
        
        q1 = np.nanpercentile(values, 25)
        q3 = np.nanpercentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return values[(values >= lower_bound) & (values <= upper_bound)]
        
    @property
    def data(self) -> dict:
        """
        Get a copy of the performance analysis data.
        The data is stored in a dictionary format.

        :return: Dictionary containing the performance analysis data.
        :rtype: dict
        """
        data = self._data.copy()

        # Remove the 'f_agn = 0.5' key if it exists
        if "f_agn = 0.5" in data:
            data.pop("f_agn = 0.5")

        return data
    
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
    
    @property
    def model_name(self) -> str:
        """
        Get the model name.
        """
        return self._model_name
    
    def get_dict(self, remove_outliers: bool = True) -> pd.DataFrame:
        data = self.data # key: f_agn = 0.?*, value: {count: int, evaluation: dict}

        data_dict = {}
        for f_agn, value in data.items():
            count, evaluation = value.get("count", 0), value.get("evaluation", {})
            for measure, values in evaluation.items():
                data_dict.setdefault(measure, []).extend(values)

        return data_dict
    
    def get_df(
            self,
            sort_by_fagn: bool = True,
            remove_outlier: bool = True,
            skip_outlier_removal: list[str] = ["Reconstruction Loss"]
        ) -> pd.DataFrame:
        """
        Get the performance analysis DataFrame.
        The rows are the AGN fractions and the columns are the performance metrics.

        :param sort_by_fagn: Whether to sort the DataFrame by AGN fraction.
        :type sort_by_fagn: bool
        :return: DataFrame containing the performance analysis.
        :rtype: pd.DataFrame
        """
        data = self.data

        df = pd.DataFrame({})

        for key, value in data.items(): # (f = ?*, {"count": ?, "evaluation": {}})
            count, evaluation = value["count"], value["evaluation"]

            # Remove the fluxes from the evaluation dictionary
            filtered_evaluation = {
                k: v
                for k, v in evaluation.items()
                if k not in self._fluxes_keys
            }
            
            for k, v in filtered_evaluation.items(): # (metric: value)
                v = np.array(v)

                if remove_outlier and k not in skip_outlier_removal:
                    # Replace inf and -inf with NaN (NumPy version)
                    v = np.where(np.isinf(v), np.nan, v)

                    # Remove outliers using the IQR method
                    v = self._remove_outliers(v)

                    # Remove NaN values
                    v = v[~np.isnan(v)] # Remove NaN values

                # Take the average
                v = v.mean() if len(v) > 0 else float('nan')

                df.loc[f"{key} ({count})", k] = v # row: `f_agn = ?* (?*)`; column: metric

        if sort_by_fagn:
            df.sort_index(inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf and -inf with NaN
        df[df > 1000] = float('nan') # Replace values greater than 1000 with NaN
        df[df < -1000] = float('nan') # Replace values less than -1000 with NaN
        df = df.round(3)

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

    def get_flux_dict(self):
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
        data = self.data

        flux_data = {}
        for key, value in data.items():  # (f = ?*, {"count": ?, "evaluation": {}})
            count, evaluation = value["count"], value["evaluation"]
            flux_entry = {k: v for k, v in evaluation.items() if k in self._fluxes_keys}
            flux_entry["count"] = count
            flux_data[key] = flux_entry

        # Sort by f = ?*
        flux_data = dict(sorted(flux_data.items(), key=lambda x: float(x[0].split("=")[1].strip())))
        return flux_data  # key: f = ?*; value: {"count": ?, ...}

    def get_all_fluxes_np(self):
        """
        Get all the fluxes from the performance analysis dictionary as NumPy arrays.
        The fluxes are concatenated across all AGN fractions.
        The structure is as follows:
            - real_psf_fluxes: np.ndarray of real PSF fluxes
            - predicted_psf_fluxes: np.ndarray of predicted PSF fluxes
            - real_gal_fluxes: np.ndarray of real galaxy fluxes
            - predicted_gal_fluxes: np.ndarray of predicted galaxy fluxes

        :return: Tuple containing the real and predicted fluxes for PSF and galaxy images.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        flux_data = self.get_flux_dict()

        # Initialize lists to store the fluxes for each AGN fraction
        real_psf_fluxes_list = []
        predicted_psf_fluxes_list = []
        real_gal_fluxes_list = []
        predicted_gal_fluxes_list = []

        # Iterate over the flux data and collect the fluxes
        for key, value in flux_data.items():
            real_psf_fluxes_list.append(value.get("real_psf_fluxes", np.array([])))
            predicted_psf_fluxes_list.append(value.get("predicted_psf_fluxes", np.array([])))
            real_gal_fluxes_list.append(value.get("real_gal_fluxes", np.array([])))
            predicted_gal_fluxes_list.append(value.get("predicted_gal_fluxes", np.array([])))

        # Concatenate the fluxes from all AGN fractions
        # If the list is empty, return an empty array
        real_psf_fluxes = np.concatenate(
            real_psf_fluxes_list, axis=0
        ) if real_psf_fluxes_list else np.array([])

        predicted_psf_fluxes = np.concatenate(
            predicted_psf_fluxes_list, axis=0
        ) if predicted_psf_fluxes_list else np.array([])

        real_gal_fluxes = np.concatenate(
            real_gal_fluxes_list, axis=0
        ) if real_gal_fluxes_list else np.array([])

        predicted_gal_fluxes = np.concatenate(
            predicted_gal_fluxes_list, axis=0
        ) if predicted_gal_fluxes_list else np.array([])
        
        
        info = "Retrieved fluxes:"
        info += f"\nReal PSF Fluxes: {real_psf_fluxes.shape}"
        info += f"\nPredicted PSF Fluxes: {predicted_psf_fluxes.shape}"
        info += f"\nReal Galaxy Fluxes: {real_gal_fluxes.shape}"
        info += f"\nPredicted Galaxy Fluxes: {predicted_gal_fluxes.shape}"
        print_box(info)

        return real_psf_fluxes, predicted_psf_fluxes, real_gal_fluxes, predicted_gal_fluxes
    
    def summarize_metrics_performance(
            self,
            remove_outlier: bool = True,
            skip_outlier_removal: list[str]|None = ["Reconstruction Loss"]
        ) -> pd.DataFrame:
        """
        Summarize the performance of the model by calculating the weighted average
        of the metrics for each AGN fraction. The weighted average is calculated
        using the counts of images in each AGN fraction bin.
        The resulting DataFrame will have the metrics as rows and the model name as the column.
        If the DataFrame is empty, it will return an empty DataFrame.

        :param remove_outlier: Whether to remove outliers from the metrics.
        If True, outliers will be removed using the IQR method.
        :type remove_outlier: bool
        :param skip_outlier_removal: List of metrics to skip outlier removal for.
        Only applicable if `remove_outlier` is True.
        :type skip_outlier_removal: list[str]
        :return: DataFrame containing the weighted average of the metrics for each AGN fraction.
        :rtype: pd.DataFrame
        """
        if remove_outlier is False and skip_outlier_removal is not None:
            AttributeMisuseWarning().warn(
                "If `remove_outlier` is False, `skip_outlier_removal` will be ignored."
            )
        
        df = self.get_df(
            remove_outlier=remove_outlier,
            skip_outlier_removal=skip_outlier_removal
        ) # row: `f_agn = ?* (?*)`; column: metric

        # If the DataFrame is empty, return an empty DataFrame
        if df.empty:
            return pd.DataFrame()
        
        # Get rows and columns
        index = df.index.tolist() # `f_agn = ?* (counts)`
        metrics = df.columns.tolist()

        # Get all the counts from the index
        counts = [int(f_agn.split("(")[-1].strip(")")) for f_agn in index]

        master_df = pd.DataFrame({})
        for m in metrics:
            values = df[m].values

            # Get the indecies of the nana values
            nan_indices = np.isnan(values)

            # Slice the values to remove the NaN values and
            # the corresponding counts. Perform a weighted average
            # of the values using the counts.
            values = values[~nan_indices]
            counts_filtered = np.array(counts)[~nan_indices]
            weighted_average = np.sum(values * counts_filtered) / np.sum(counts_filtered)

            master_df.loc[m, self._model_name] = weighted_average

        return master_df
    