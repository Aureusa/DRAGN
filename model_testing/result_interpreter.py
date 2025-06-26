from utils import print_box
import pandas as pd
import numpy as np

from model_testing.performance_analysis import PAdict
from scipy.stats import pearsonr


class ResultInterpreter:
    """
    Class to interpret the performance analysis of a model.
    It contains methods to interpret the performance analysis and compare
    the performance of different models.
    """
    def frf_flux_correlation(
            self,
            data: PAdict,
            model_name: str,
            verbose: bool = True,
            latex: bool = False,
            true_fluxes_keys: list[str] = ["real_psf_fluxes", "real_gal_fluxes"],
            predicted_fluxes: list[str] = ["predicted_psf_fluxes", "predicted_gal_fluxes"]
        ):
        data = data.get_dict() # key: metric/flux, value: list of values

        # Extract the real fluxes
        real_psf_fluxes = data.get(true_fluxes_keys[0], [])
        real_gal_fluxes = data.get(true_fluxes_keys[1], [])

        if len(real_psf_fluxes) == 0 or len(real_gal_fluxes) == 0:
            raise ValueError("Real fluxes are not available in the data. Please check the data structure.")
        
        # Extract the predicted fluxes
        predicted_psf_fluxes = data.get(predicted_fluxes[0], [])
        predicted_gal_fluxes = data.get(predicted_fluxes[1], [])

        if len(predicted_psf_fluxes) == 0 or len(predicted_gal_fluxes) == 0:
            raise ValueError("Predicted fluxes are not available in the data. Please check the data structure.")

        # Extract the FRF metrics
        frf_psf = data.get("FRF PSF", [])
        frf_gal = data.get("FRF", [])

        if len(frf_psf) == 0 or len(frf_gal) == 0:
            raise ValueError("FRF metrics are not available in the data. Please check the data structure.")
        
        # Create a DataFrame with the fluxes and FRF metrics
        df = pd.DataFrame({
            "Real PSF Flux": real_psf_fluxes,
            "Predicted PSF Flux": predicted_psf_fluxes,
            "Real Galaxy Flux": real_gal_fluxes,
            "Predicted Galaxy Flux": predicted_gal_fluxes,
            "FRF PSF": frf_psf,
            "FRF Galaxy": frf_gal
        })

        # Remove the rows where the FRF metrics are <= -20 and => 20
        # as well as galaxies with negative fluxes
        df = df[(df["Real PSF Flux"] > 0)]
        df = df[(df["Real Galaxy Flux"] > 0)]
        df = df[(df["FRF PSF"] >= -20) & (df["FRF PSF"] <= 20)]
        df = df[(df["FRF Galaxy"] >= -20) & (df["FRF Galaxy"] <= 20)]

        # Remove rows with NaN values
        df = df.dropna()

        # Split the DataFrame into two DataFrames: one for PSF and one for Galaxy
        df_psf = df[["Real PSF Flux", "Predicted PSF Flux", "FRF PSF"]]
        df_gal = df[["Real Galaxy Flux", "Predicted Galaxy Flux", "FRF Galaxy"]]

        # Remove the upper 5% of the fluxes
        upper_limit_psf = np.percentile(df["Real PSF Flux"], 99.5)
        upper_limit_gal = np.percentile(df["Real Galaxy Flux"], 99.5)
        df_psf = df_psf[df_psf["Real PSF Flux"] <= upper_limit_psf]
        df_gal = df_gal[df_gal["Real Galaxy Flux"] <= upper_limit_gal]

        # Bin by Real PSF Flux into 20 bins
        df_psf['flux_bin'] = pd.cut(df_psf["Real PSF Flux"], bins=20)
        df_gal['flux_bin'] = pd.cut(df_gal["Real Galaxy Flux"], bins=20)

        # Group by the flux bin and calculate the mean of the FRF metrics
        df_psf = df_psf.groupby('flux_bin').mean().reset_index()
        df_gal = df_gal.groupby('flux_bin').mean().reset_index()

        # Sort by fluxes
        df_psf = df_psf.sort_values(by="Real PSF Flux")
        df_gal = df_gal.sort_values(by="Real Galaxy Flux")

        # Remove the first row of df_psf
        if not df_psf.empty:
            df_psf = df_psf.iloc[1:].reset_index(drop=True)

        if verbose:
            print_box(f"Model: {model_name} - PSF Flux Correlation")
            print(df_psf)
            print_box(f"Model: {model_name} - Galaxy Flux Correlation")
            print(df_gal)

        if latex:
            print_box("PSF Flux Correlation in LaTeX format:")
            latex_str_psf = df_psf.to_latex(
                index=False,
                float_format="%.3f",
                caption=f"PSF Flux Correlation for {model_name}",
                label=f"tab:psf_flux_correlation_{model_name.lower().replace(' ', '_')}"
            )
            print(latex_str_psf)

            print_box("Galaxy Flux Correlation in LaTeX format:")
            latex_str_gal = df_gal.to_latex(
                index=False,
                float_format="%.3f",
                caption=f"Galaxy Flux Correlation for {model_name}",
                label=f"tab:gal_flux_correlation_{model_name.lower().replace(' ', '_')}"
            )
            print(latex_str_gal)

        return df_psf, df_gal
        
    def get_correlation_matrix(
            self,
            data: PAdict,
            model_name: str,
            verbose: bool = True,
            latex: bool = False,
            flux_columns2keep: list[str] = ["real_psf_fluxes", "real_gal_fluxes"],
            flux_columns2drop: list[str] = ["predicted_psf_fluxes", "predicted_gal_fluxes"]
        ) -> pd.DataFrame:
        data = data.data # key: f_agn = 0.?*, value: {count: int, evaluation: dict}

        data_dict = {}
        for f_agn, value in data.items():
            count, evaluation = value.get("count", 0), value.get("evaluation", {})
            for measure, values in evaluation.items():
                if count == len(values):
                    if measure not in flux_columns2keep:
                        values = [np.nan if v > 1000 or v < -1000 else v for v in values]
                    data_dict.setdefault(measure, []).extend(values)

        # Drop the flux columns that are in flux_columns2drop
        for col in flux_columns2drop:
            if col in data_dict:
                data_dict.pop(col)

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))

        # Outlier mask: True if not an outlier, False if outlier
        mask = pd.DataFrame(True, index=df.index, columns=df.columns)
        for col in df.columns:
            if col in flux_columns2keep:
                continue
            arr = df[col].values.astype(float)
            if np.sum(~np.isnan(arr)) > 0:
                q1 = np.nanpercentile(arr, 25)
                q3 = np.nanpercentile(arr, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask[col] = (arr >= lower) & (arr <= upper)


        # Keep only rows where all metrics are not outliers
        df_filtered = df[mask.all(axis=1)].dropna(axis=0, how='any')

        # Now you can safely compute correlations
        corr = df_filtered.corr(method="spearman")

        # Compute p-values for the correlations
        pvals = pd.DataFrame(np.ones(corr.shape), columns=corr.columns, index=corr.index)
        for i in corr.columns:
            for j in corr.columns:
                if i == j:
                    pvals.loc[i, j] = 0.0
                else:
                    # Drop NaNs for pairwise calculation
                    x = df_filtered[i]
                    y = df_filtered[j]
                    mask = x.notna() & y.notna()
                    if mask.sum() > 1:
                        _, p = pearsonr(x[mask], y[mask])
                        pvals.loc[i, j] = p
                    else:
                        pvals.loc[i, j] = np.nan

        if verbose:
            print(f"Model: {model_name}")
            print(corr)
            print(pvals)

        return corr, pvals

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
        df = data.get_df(sort_by_fagn=True)

        # Depricated:
        # Drop the column "PSF-MSE Loss" if it exists
        if "PSF-MSE Loss" in df.columns:
            df = df.drop(columns="PSF-MSE Loss")
        
        if verbose:
            print(f"Model: {model_name}")
            print(df)

        if latex:
            print_box("Performance analysis in LaTeX format:")
            latex_str = df.to_latex(
                index=True,
                float_format="%.3f",
                caption=f"Performance Analysis of {model_name}",
                label=f"tab:performance_analysis_{model_name.lower().replace(' ', '_')}"
            )
            # Replace all occurrences of 'table' with 'table*', insert \centering after \begin{table}, and replace 'f_agn' with '$f_{AGN}$'
            latex_lines = latex_str.splitlines()
            new_lines = []
            for line in latex_lines:
                # Replace all 'table' with 'table*'
                line = line.replace("table", "table*")
                # Insert \centering after \begin{table*}
                if line.strip().startswith(r"\begin{table*}"):
                    new_lines.append(line)
                    new_lines.append(r"\centering")
                    continue
                # Replace 'f_agn' with '$f_{AGN}$'
                line = line.replace("f_agn", r"$f_{AGN}$")
                new_lines.append(line)
            print("\n".join(new_lines))

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
            df = data.summarize_metrics_performance()

            # Concatenate the single-column DataFrame to the master DataFrame
            master_df = pd.concat([master_df, df], axis=1)

        # Sort the DataFrame by the "FRF" column in descending order if it exists
        if "FRF" in master_df.index:
            master_df = master_df.loc[:, master_df.loc["FRF"].abs().sort_values().index]
            master_df = master_df.round(3)

        if verbose:
            print_box("Performance comparison:")
            print(master_df.to_string(index=True))

        if latex:
            print_box("Performance comparison in LaTeX format:")
            latex_str = master_df.to_latex(
                index=True,
                float_format="%.3f",
                caption="Performance Analysis of Different Models",
                label="tab:performance_analysis_summary"
            )

            # Replace \begin{table} with \begin{table*} and insert \centering after it
            latex_lines = latex_str.splitlines()
            for i, line in enumerate(latex_lines):
                if line.strip().startswith(r"\begin{table}"):
                    latex_lines[i] = line.replace(r"table", r"table*")
                    latex_lines.insert(i + 1, r"\centering")
                if line.strip().startswith(r"\end{table}"):
                    latex_lines[i] = line.replace(r"table", r"table*")          
            print("\n".join(latex_lines))

        return master_df
    