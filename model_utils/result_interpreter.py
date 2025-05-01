from utils import print_box
import pandas as pd

from model_utils.performance_analysis import PAdict


class ResultInterpreter:
    """
    Class to interpret the performance analysis of a model.
    It contains methods to interpret the performance analysis and compare
    the performance of different models.
    """
    def interpret_performance_analyis(
            self,
            data: PAdict,
            model_name: str,
            verbose: bool = True,
            latex: bool = False
        ) -> pd.DataFrame:
        """
        Interpret the performance analysis of a model. It creates a dataframe
        containing the performance metrics for each AGN fraction.

        :param data: Performance analysis dictionary for the model.
        :type data: PAdict
        :param model_name: Name of the model.
        :type model_name: str
        :param verbose: Whether to print the performance analysis.
        :type verbose: bool
        :param latex: Whether to print the performance analysis in LaTeX format.
        :type latex: bool
        :return: DataFrame containing the performance analysis. The rows are
        the AGN fractions and the columns are the performance metrics.
        :rtype: pd.DataFrame
        """
        df = data.get_df()

        # Sort the DataFrame by the "PSNR" column in descending order
        df = df.sort_values(by="PSNR", ascending=False)
        df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        df = df.round(3)

        # Depricated:
        # Drop the column "PSF-MSE Loss" if it exists
        if "PSF-MSE Loss" in df.columns:
            df = df.drop(columns="PSF-MSE Loss")
        
        if verbose:
            print(f"Model: {model_name}")
            print(df)

        if latex:
            print_box("Performance analysis in LaTeX format:")
            print(df.to_latex(
                index=True,
                float_format="%.3f",
                caption=f"Performance Analysis of {model_name}",
                label=f"tab:performance_analysis_{model_name.lower().replace(' ', '_')}"
            ))

        return df

    def compare_model_performance(
            self,
            data_list: list[PAdict],
            verbose: bool = True,
            latex: bool = False
        ) -> pd.DataFrame:
        """
        Compare the performance analysis of different models. It creates a dataframe
        containing the average performance metrics for each model. The performance
        analysis of each model MUST have the same structure and metrics.

        :param data_list: List of performance analysis dictionaries for different models.
        :type data_list: list[PAdict]
        :param verbose: Whether to print the performance analysis.
        :type verbose: bool
        :param latex: Whether to print the performance analysis in LaTeX format.
        :type latex: bool
        :return: DataFrame containing the performance analysis comparison.
        :rtype: pd.DataFrame
        """
        master_df = pd.DataFrame({})
        for data in data_list:
            df = data.summarize_performance()

            # Concatenate the single-column DataFrame to the master DataFrame
            master_df = pd.concat([master_df, df], axis=1)

        # Sort the DataFrame by the "PSNR" column in descending order if it exists
        if "PSNR" in master_df.index:
            master_df = master_df.sort_values(by="PSNR", axis=1, ascending=False)
            master_df.round(3)
        
        # Depricated:
        # Drop the index "PSF-MSE Loss" if it exists
        if "PSF-MSE Loss" in master_df.index:
            master_df = master_df.drop(index="PSF-MSE Loss")

        if verbose:
            print_box("Performance comparison:")
            print(master_df.to_string(index=True))

        if latex:
            print_box("Performance comparison in LaTeX format:")
            print(master_df.to_latex(
                index=True,
                float_format="%.3f",
                caption="Performance Analysis of Different Models",
                label="tab:performance_analysis_summary"
            ))

        return master_df
    