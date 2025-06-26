import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import re
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

from model_testing.performance_analysis import PAdict
from model_testing.metrics import get_metrics
from loggers_utils import log_execution
from utils import print_box
from utils_utils.validation import validate_numpy_array, validate_list
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import shift


class Plotter:
    def plot_loss(
            self,
            train_loss: list[float],
            val_loss: list[float],
            best_val_loss: list[str],
            filename: str,
            data_folder: str
        ) -> None:
        """
        Plot the training and validation loss.

        :param train_loss: List of training loss values.
        :type train_loss: list[float]
        :param val_loss: List of validation loss values.
        :type val_loss: list[float]
        :param best_val_loss: Best validation loss value.
        :type best_val_loss: float
        :param filename: Filename to save the plot.
        :type filename: str
        :param data_folder: Folder to save the plot.
        :type data_folder: str
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label=f"Validation Loss; Best: {best_val_loss:.4f}")
        plt.title("Train and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(data_folder ,f"{filename}_loss.png"))

        print_box(f"Loss plot for '{filename}' saved successfully!")

    def plot_two_histograms(
        self,
        arr1: np.ndarray,
        arr2: np.ndarray,
        label1: str,
        label2: str,
        title: str,
        filename: str,
        data_folder: str,
        bins: int = 50,
        save: bool = True
        ) -> None:
        """
        Plot two histograms on top of each other for comparison.

        :param arr1: First array of values.
        :type arr1: np.ndarray
        :param arr2: Second array of values.
        :type arr2: np.ndarray
        :param label1: Label for the first array.
        :type label1: str
        :param label2: Label for the second array.
        :type label2: str
        :param title: Title of the plot.
        :type title: str
        :param filename: Filename for saving the plot.
        :type filename: str
        :param data_folder: Folder to save the plot.
        :type data_folder: str
        :param bins: Number of bins for the histogram.
        :type bins: int
        :param save: Whether to save the plot or show it.
        :type save: bool
        """
        validate_numpy_array(arr1, ndim=1)
        validate_numpy_array(arr2, ndim=1)

        # Remove the 99.5 percentile outliers
        # arr1 = arr1[arr1 < np.percentile(arr1, 99)]
        # arr2 = arr2[arr2 < np.percentile(arr2, 99)]

        # Histogram of the pixel values
        hist1 = np.histogram(arr1, bins=bins)
        hist2 = np.histogram(arr2, bins=bins)

        # Compute the FFT of the arrays
        arr1_fft = np.fft.fft(arr1)
        arr2_fft = np.fft.fft(arr2)

        # Compute the power spectrum
        arr1_power = np.abs(arr1_fft)**2
        arr2_power = np.abs(arr2_fft)**2

        # Remove the 99.5 percentile outliers from the power spectrum
        # arr1_power = arr1_power[arr1_power < np.percentile(arr1_power, 99)]
        # arr2_power = arr2_power[arr2_power < np.percentile(arr2_power, 99)]

        # Compute the histograms of the power spectrum
        hist1_power = np.histogram(arr1_power, bins=bins)
        hist2_power = np.histogram(arr2_power, bins=bins)

        # Angle
        arr1_angle = np.angle(arr1_fft)
        arr2_angle = np.angle(arr2_fft)

        # Compute the histograms of the angles
        hist1_angle = np.histogram(arr1_angle, bins=bins)
        hist2_angle = np.histogram(arr2_angle, bins=bins)

        fig, ax = plt.subplots(figsize=(18, 6))

        # Histogram of values
        ax.hist(hist1[1][:-1], bins=hist1[1], weights=hist1[0], alpha=0.6, label=label1, color='blue', density=True)
        ax.hist(hist2[1][:-1], bins=hist2[1], weights=hist2[0], alpha=0.6, label=label2, color='orange', density=True)
        ax.set_title(f"{title} (Values)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        if save:
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
                print_box(f"Data folder '{data_folder}' created successfully!")
            filepath = os.path.join(data_folder, f"{filename}.png")
            plt.savefig(filepath)
            print_box(f"Histogram '{filename}.png' saved successfully in {data_folder}!")
        else:
            plt.show()
            plt.close()

    def make_frf_flux_correlation_plot(
            self,
            df_psf_list: list[pd.DataFrame],
            df_gal_list: list[pd.DataFrame],
            model_names: list[str],
            filename: str,
            data_folder: str
        ) -> None:
        """
        Plot FRF vs Real Flux for PSF and Galaxy, similar to trend_plot style.

        :param df_psf: DataFrame with PSF flux and FRF columns.
        :type df_psf: pd.DataFrame
        :param df_gal: DataFrame with Galaxy flux and FRF columns.
        :type df_gal: pd.DataFrame
        :param filename: Filename to save the plots.
        :type filename: str
        :param data_folder: Folder to save the plots.
        :type data_folder: str
        """
        plt.figure(figsize=(10, 6))

        for i in range(len(model_names)):
            df_psf = df_psf_list[i]
            df_gal = df_gal_list[i]

            # Validate DataFrames
            if not isinstance(df_psf, pd.DataFrame) or not isinstance(df_gal, pd.DataFrame):
                raise TypeError("Both df_psf and df_gal should be pandas DataFrames.")
            if "FRF PSF" not in df_psf.columns or "FRF Galaxy" not in df_gal.columns:
                raise ValueError("DataFrames must contain 'FRF PSF' and 'FRF Galaxy' columns.")
            if "Real PSF Flux" not in df_psf.columns or "Real Galaxy Flux" not in df_gal.columns:
                raise ValueError("DataFrames must contain 'Real PSF Flux' and 'Real Galaxy Flux' columns.")

            # Extract data
            frf_psf = df_psf["FRF PSF"]
            real_psf_flux = df_psf["Real PSF Flux"]

            # Plot each model's FRF vs Real Flux
            plt.plot(real_psf_flux, frf_psf, marker='o', linestyle='-', label=model_names[i])

        plt.axhline(0, color='red', linestyle='--', label='Perfect FRF PSF')
        plt.xlabel("Real PSF Flux (Jy)")
        plt.ylabel("FRF")
        plt.title("FRF PSF vs Real PSF Flux")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_folder, f"frf_psf_vs_flux.png"))
        print_box(f"FRF (PSF) vs Real PSF Flux plot saved as 'frf_psf_vs_flux.png' in {data_folder}!")
        plt.close()

        # Galaxy plot
        plt.figure(figsize=(10, 6))
        for i in range(len(model_names)):
            df_psf = df_psf_list[i]
            df_gal = df_gal_list[i]

            # Validate DataFrames
            if not isinstance(df_psf, pd.DataFrame) or not isinstance(df_gal, pd.DataFrame):
                raise TypeError("Both df_psf and df_gal should be pandas DataFrames.")
            if "FRF PSF" not in df_psf.columns or "FRF Galaxy" not in df_gal.columns:
                raise ValueError("DataFrames must contain 'FRF PSF' and 'FRF Galaxy' columns.")
            if "Real PSF Flux" not in df_psf.columns or "Real Galaxy Flux" not in df_gal.columns:
                raise ValueError("DataFrames must contain 'Real PSF Flux' and 'Real Galaxy Flux' columns.")

            # Extract data
            frf_gal = df_gal["FRF Galaxy"]
            real_gal_flux = df_gal["Real Galaxy Flux"]

            # Plot each model's FRF vs Real Flux
            plt.plot(real_gal_flux, frf_gal, marker='o', linestyle='-', label=model_names[i])

        plt.axhline(0, color='red', linestyle='--', label='Perfect FRF')
        plt.xlabel("Real Galaxy Flux (Jy)")
        plt.ylabel("FRF (Galaxy)")
        plt.title("FRF (Galaxy) vs Real Galaxy Flux")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_folder, f"frf_gal_vs_flux.png"))
        print_box(f"FRF (Galaxy) vs Real Galaxy Flux plot saved as 'frf_gal_vs_flux.png' in {data_folder}!")
        plt.close()
        

    def plot_correlation_heatmap(
            self,
            corr: pd.DataFrame,
            filename: str,
            data_folder: str,
            cmap: str = "coolwarm",
            annot: bool = True,
            save: bool = True
        ) -> None:
        """
        Plot a correlation heatmap from a DataFrame produced by df.corr().

        :param corr: Correlation matrix DataFrame (output of df.corr()).
        :type corr: pd.DataFrame
        :param filename: Filename to save the plot.
        :type filename: str
        :param data_folder: Folder to save the plot.
        :type data_folder: str
        :param cmap: Colormap for the heatmap.
        :type cmap: str
        :param annot: Whether to annotate the heatmap with correlation values.
        :type annot: bool
        :param save: Whether to save the plot or show it.
        :type save: bool
        """
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, label="Correlation coefficient")
        plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
        if annot:
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        if save:
            filepath = os.path.join(data_folder, f"{filename}.png")
            plt.savefig(filepath)
            print_box(f"Correlation heatmap '{filename}.png' saved successfully in {data_folder}!")
        else:
            plt.show()
        plt.close()

    def grid_plot_no_model(self, images: list[np.ndarray], save: bool = False, filepath: str = os.getcwd()) -> None:
        images = np.array(images) # (B, C, H, W)
        validate_numpy_array(images, ndim=4)
        num_images = images.shape[0]

        if num_images > 36:
            images = images[:36]  # Limit to 36 images for the grid
        else:
            raise ValueError("Number of images must be at least 36 for a grid plot.")
        
        num_cols = 6
        num_rows = 6

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < num_images:
                    axes[i, j].imshow(
                        images[idx].transpose(1, 2, 0),
                        cmap='gray',
                        norm=ImageNormalize(
                            images[idx].transpose(1, 2, 0),
                            interval=PercentileInterval(99.5),
                            stretch=AsinhStretch())
                        )
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        plt.tight_layout()

        if save:
            filepath_ = os.path.join(filepath, "grid_plot_no_model.png")
            if not os.path.exists(filepath_):
                os.makedirs(os.path.dirname(filepath_), exist_ok=True)
                print_box(f"Data folder '{os.path.dirname(filepath_)}' created successfully!")
            plt.savefig(filepath_, bbox_inches='tight', dpi=200)
            print_box(f"Grid plot saved successfully as '{filepath}'!")
        else:
            plt.show()

        plt.close()

    @log_execution("Creating grid plot...", "Grid plot created successfully!")
    def grid_plot(
            self,
            sources: list[np.ndarray],
            targets: list[np.ndarray]|None,
            outputs: list[np.ndarray],
            titles: list[str],
            filename: str,
            data_folder: str,
            f_agn: list[int]|None,
            save: bool = False,
            norm: bool = True
        ) -> None:
        """
        Create a tight grid of images for report/figure, similar to pix2pix paper.
        Columns: Input | Target | Output (per model)
        Rows: Each image in the batch.
        """
        # Convert lists to numpy arrays
        # shape: (num_models, num_agn_batches, num_f_agn, width, height)
        sources_arr = np.array(sources)
        if targets is not None:
            targets_arr = np.array(targets)
            targets_arr = np.transpose(targets_arr, (1, 2, 0, 3, 4))
        outputs_arr = np.array(outputs)

        # Transpose arrays to shape (num_agn_batches, ...) for easier iteration
        # shape:
        # (num_models, num_agn_batches, num_fagn, width, height) ->
        # (num_agn_batches, num_fagn, num_models, width, height)
        sources_arr = np.transpose(sources_arr, (1, 2, 0, 3, 4))
        outputs_arr = np.transpose(outputs_arr, (1, 2, 0, 3, 4))

        if norm:
            # Normalize by dividintg by the maximum value in each image
            sources_arr = sources_arr / np.max(sources_arr, axis=(3, 4), keepdims=True)
            targets_arr = targets_arr / np.max(targets_arr, axis=(3, 4), keepdims=True) if targets is not None else None
            outputs_arr = outputs_arr / np.max(outputs_arr, axis=(3, 4), keepdims=True)

        if f_agn is None:
            NUM_B, NUM_FAGN, NUM_MODELS, W, H = sources_arr.shape
            sources_arr = sources_arr.reshape(NUM_B//4, 4, NUM_MODELS, W, H)
            outputs_arr = outputs_arr.reshape(NUM_B//4, 4, NUM_MODELS, W, H)
            targets_arr = targets_arr.reshape(NUM_B//4, 4, NUM_MODELS, W, H) if targets is not None else None

        for i in range(sources_arr.shape[0]):
            num_batch = i + 1
            # Get the images for the batch of images
            # shape: (num_fagn, num_models, width, height)
            sources = sources_arr[i]
            if targets is not None:
                targets = targets_arr[i]
            outputs = outputs_arr[i]
            
            num_fagn = sources.shape[0] # number of fagns
            num_models = sources.shape[1] # number of models

            # Prepare normalization (use target normalization for all)
            # shape
            interval = PercentileInterval(99.5)
            if targets is not None:
                norms = [ImageNormalize(targets[i, 0], interval=interval, stretch=AsinhStretch()) for i in range(num_fagn)]
                if norm:
                    norms = [ImageNormalize(sources[i, 0], interval=interval, stretch=AsinhStretch()) for i in range(num_fagn)]
            else:
                norms = []
                for i in range(num_fagn):
                    arr = sources[i, 0]
                    if arr.size == 0:
                        # Handle the empty case: skip, append None, or use a default normalization
                        norms.append(None)
                        continue
                    norms.append(ImageNormalize(arr, interval=interval, stretch=AsinhStretch()))

            # Figure: rows = num_fagn, cols = 2 + num_models
            cols = 2 + num_models if targets is not None else 1 + num_models
            ax = plt.subplots(num_fagn, cols, figsize=(3 * cols, 3 * num_fagn))[1]

            # If only one image, ax is 1D
            if num_fagn == 1:
                ax = np.expand_dims(ax, 0)
            if cols == 1:
                ax = np.expand_dims(ax, 1)

            for i in range(num_fagn):
                # Input (first model's input for each image)
                ax[i, 0].imshow(sources[i, 0], norm=norms[i], cmap="gray")
                ax[i, 0].axis("off")
                if f_agn is not None:
                    # Add f_agn text to the top left
                    ax[i, 0].text(
                        0.02, 0.08, f"f_AGN = 0.{f_agn[i]}",
                        color="white", fontsize=12, fontweight="bold",
                        ha="left", va="top", transform=ax[i, 0].transAxes,
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
                    )
                if targets is not None:
                    # Target (first model's target for each image)
                    ax[i, 1].imshow(targets[i, 0], norm=norms[i], cmap="gray")
                    ax[i, 1].axis("off")
                # Outputs for each model
                for m in range(num_models):
                    ax[i, 2 + m if targets is not None else 1 + m].imshow(
                        outputs[i, m],
                        norm=norms[i],
                        cmap="gray"
                    )
                    ax[i, 2 + m if targets is not None else 1 + m].axis("off")

            # Set column titles
            if targets is not None:
                col_titles = ["Input", "Target"] + list(titles)
            else:
                col_titles = ["Input"] + list(titles)

            for j, title in enumerate(col_titles):
                ax[0, j].set_title(title, fontsize=14, fontweight="bold")

            plt.tight_layout()
            if save:
                filepath = os.path.join(data_folder, f"{filename}_report_{num_batch}.png")
                plt.savefig(filepath, bbox_inches="tight", dpi=200)
                print_box(f"Grid plot '{filename}_report_{num_batch}.png' saved successfully in {data_folder}!")
            else:
                plt.show()
            plt.close()

    @log_execution("Making diagnostic plots...", "Diagnostic plots made successfully!")
    def diagnostic_plot(
            self,
            sources: list[np.ndarray],
            targets: list[np.ndarray],
            outputs: list[np.ndarray],
            predicted_psfs: list[np.ndarray],
            psfs: list[np.ndarray],
            titles: list[str],
            filename: str,
            data_folder: str,
            show_real_min_infered: bool = False,
            save: bool = False,
        ) -> None:
        """
        Plot the sources, targets, cleaned images, PSFs, and predicted PSFs.

        :param sources: List of source images.
        :type sources: list[np.ndarray]
        :param targets: List of target images.
        :type targets: list[np.ndarray]
        :param outputs: List of cleaned images.
        :type outputs: list[np.ndarray]
        :param predicted_psfs: List of predicted PSF images.
        :type predicted_psfs: list[np.ndarray]
        :param psfs: List of PSF images.
        :type psfs: list[np.ndarray]
        :param titles: List of titles for each model.
        :type titles: list[str]
        :param norms: List of normalization objects for each image.
        :type norms: list[ImageNormalize]
        :param filename: Filename for saving the plot.
        :type filename: str
        :param data_folder: Folder to save the plot.
        :type data_folder: str
        :param show_real_min_infered: Whether to show the real minus inferred image.
        :type show_real_min_infered: bool
        :param save: Whether to save the plot or show it.
        :type save: bool
        :raises ValueError: If the input arrays do not have the same shape.
        """
        # Convert lists to numpy arrays
        # shape: (num_models, num_images, num_f_agn, width, height)
        sources_arr = np.array(sources)
        targets_arr = np.array(targets)
        outputs_arr = np.array(outputs)
        predicted_psfs_arr = np.array(predicted_psfs)
        psfs_arr = np.array(psfs)

        # Check if all arrays have the same shape
        self._compatibility_check(sources_arr, targets_arr, outputs_arr, predicted_psfs_arr, psfs_arr)

        M, N, FAGN, W, H = sources_arr.shape

        # Reshape to (num_models, num_images, width, height)
        new_shape = (M, N * FAGN, W, H)
        sources_arr = sources_arr.reshape(new_shape)
        targets_arr = targets_arr.reshape(new_shape)
        outputs_arr = outputs_arr.reshape(new_shape)
        predicted_psfs_arr = predicted_psfs_arr.reshape(new_shape)
        psfs_arr = psfs_arr.reshape(new_shape)

        for i in range(sources_arr.shape[1]):
            # Get the images for the batch of images
            sources = sources_arr[:, i, :, :] # (num_models, width, height)
            targets = targets_arr[:, i, :, :]
            outputs = outputs_arr[:, i, :, :]
            predicted_psfs = predicted_psfs_arr[:, i, :, :]
            psfs = psfs_arr[:, i, :, :]

            # Create the plot
            self._make_plot(
                sources=sources,
                targets=targets,
                outputs=outputs,
                predicted_psfs=predicted_psfs,
                psfs=psfs,
                titles=titles,
                count=i,
                filename=filename,
                data_folder=data_folder,
                show_real_min_infered=show_real_min_infered,
                save=save
            )

    def make_trend_plots_fluxes(
            self,
            pa_list: list[PAdict],
            data_folder: str,
            save: bool = True
        ) -> None:
        validate_list(pa_list, PAdict)

        # Sort the DataFrames by index (f_agn) and check
        # if all dataframes have the same rows and columns
        sorted_dfs = []
        model_names = []
        for i, pa in enumerate(pa_list):
            # Perform sanity check to ensure all DataFrames have the same index and columns
            if pa != pa_list[0]:
                raise IndexError("DataFrames have different index or columns. Please check the input data.")

            # Sort the DataFrame by index (f_agn)
            df = pa.get_df(sort_by_fagn=True)
            sorted_dfs.append(df)
            model_names.append(pa.model_name)

    # DEPRICATED: remove later
    def make_trend_plots_fagn(
            self,
            pa_list: list[PAdict],
            model_names: list[str],
            data_folder: str,
            save: bool = True
        ) -> None:
        """
        Make trend plot for any number of metrics and any number of models. You
        can pass any number of performance analysis DataFrames and the function
        will create a trend plot for each metric in the DataFrames.
        The trend plot will show the trend of the metric with respect to the AGN
        fraction (f_agn) for each model.

        :param pa_list: List of performance analysis dicts.
        :type pa_list: list[PAdict]
        :param data_folder: Folder to save the trend plots.
        :type data_folder: str
        :raises IndexError: If the DataFrames have different index or columns.
        """
        validate_list(pa_list, PAdict)

        # Sort the DataFrames by index (f_agn) and check
        # if all dataframes have the same rows and columns
        sorted_dfs = []
        for i, pa in enumerate(pa_list):
            # Perform sanity check to ensure all DataFrames have the same index and columns
            if pa != pa_list[0]:
                raise IndexError("DataFrames have different index or columns. Please check the input data.")

            # Sort the DataFrame by index (f_agn)
            df = pa.get_df(sort_by_fagn=True)
            sorted_dfs.append(df)

        # Get the rows (indecies) and columns (metrics) from the first DataFrame
        # they should be the same for all DataFrames       
        indecies = sorted_dfs[0].index
        metrics = get_metrics(sorted_dfs[0].columns)

        # Create a figure for each metric
        for m in metrics:
            plt.figure(figsize=(10, 6))

            metric = str(m)
            for i, df in enumerate(sorted_dfs): # Iterate through the sorted performance analysis DataFrames
                y = df[metric]

                # DEPRICATED: remove later
                y = np.where(y < -100, np.nan, y)

                # DPERICATED: remove later
                if metric == "Centroid Error":
                    y[0] = np.nan     

                # Find the index for the nan values
                nan_indices = np.where(np.isnan(y))[0]

                # Remove the nan values from the data
                metric_values = np.delete(y, nan_indices)
                agn_fracs = np.delete(indecies, nan_indices)

                # Make the plot more readable if agn_frac is too large
                if len(agn_fracs) > 25:
                    frac = len(agn_fracs) // 25
                    agn_fracs = agn_fracs[::frac]
                    metric_values = metric_values[::frac]
                
                # Convert to percentage
                agn_fracs = [f"{int(float(f.split(' = ')[1].split(' ')[0]) * 100)}%" for f in agn_fracs]

                plt.plot(agn_fracs, metric_values, marker='o', label=model_names[i])

            y_label = metric

            m_metadata = m.get_metadata()
            unit = m_metadata.get('unit', None)
            if unit is not None:
                y_label += f" ({unit})"
            
            best_value = m_metadata.get('best_value', None)
            if best_value is not None:
                plt.axhline(y=best_value, color='red', linestyle='--', label=f'Perfect {metric}')

            plt.xlabel("AGN Fraction (f_agn - %)")
            plt.ylabel(y_label)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()

            if save:
                filepath = os.path.join(data_folder, f"{metric.lower().replace(' ', '_')}_trend_plot.png")
                plt.savefig(filepath)
                print_box(f"Trend plot for '{metric}' saved successfully!")
            else:
                plt.show()
            plt.close()

    def flux_histogram(self, fluxes: np.ndarray, title: str, filename: str, data_folder: str, bins: int = 50, save: bool = True) -> None:
        """
        Create a histogram of fluxes.

        :param fluxes: Array of fluxes.
        :type fluxes: np.ndarray
        :param filename: Filename for saving the histogram.
        :type filename: str
        :param data_folder: Folder to save the histogram.
        :type data_folder: str
        :param bins: Number of bins for the histogram.
        :type bins: int
        """
        validate_numpy_array(fluxes, ndim=1)

        if os.path.exists(data_folder) is False:
            os.makedirs(data_folder)
            print_box(f"Data folder '{data_folder}' created successfully!")

        plt.figure(figsize=(10, 6))
        plt.hist(fluxes, bins=bins, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel("Flux (Jy)")
        plt.ylabel("Count")
        plt.grid(True)

        if save:
            filepath = os.path.join(data_folder, f"{filename}.png")
            plt.savefig(filepath)
            print_box(f"Flux histogram '{filename}.png' saved successfully in {data_folder}!")
        else:
            plt.show()
        
        plt.close()

    @log_execution("Creating 2D histogram...", "2D histogram created successfully!")
    def make_2d_histogram(
            self,
            real_fluxes: np.ndarray,
            predicted_fluxes: np.ndarray,
            histogram_filename: str,
            data_folder: str,
            x_label: str = "Real Flux",
            y_label: str = "Predicted Flux",
            x_y_lim: int|None = None,
            title: str = "Real vs Predicted Flux",
            save: bool = True
        ) -> None:
        """
        Create a 2D histogram of real vs predicted fluxes.

        :param real_fluxes: Array of real fluxes. The arrays should
        be the same length and same shape of (n,).
        :type real_fluxes: np.ndarray
        :param predicted_fluxes: Array of predicted fluxes. The arrays should
        be the same length and same shape of (n,).
        :type predicted_fluxes: np.ndarray
        :param histogram_filename: Filename for saving the histogram.
        :type histogram_filename: str
        :param data_folder: Folder to save the histogram.
        :type data_folder: str
        :param x_label: Label for the x-axis.
        :type x_label: str
        :param y_label: Label for the y-axis.
        :type y_label: str
        :param x_y_lim: Limit for the x and y axes.
        If None, the limits are set to the min and max of the fluxes.
        :type x_y_lim: int|None
        """
        validate_numpy_array(real_fluxes, ndim=1)
        validate_numpy_array(predicted_fluxes, ndim=1)

        # Create a copy of the plasma colormap and set the lowest value to white
        plasma = cm.get_cmap('plasma', 256)
        newcolors = plasma(np.linspace(0, 1, 256))
        newcolors[0, :] = np.array([1, 1, 1, 1])  # RGBA for white
        white_plasma = mpl.colors.ListedColormap(newcolors)

        # Create a 2D histogram
        hist, xedges, yedges = np.histogram2d(real_fluxes, predicted_fluxes, bins=35)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Apply logarithmic scaling to the histogram
        hist = np.log1p(hist)  # log1p avoids log(0)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the histogram
        im = ax.imshow(hist.T, extent=extent, origin='lower', aspect='auto', cmap=white_plasma)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Logarithmic Count")

        # Add a 1:1 line
        min_flux = min(real_fluxes.min(), predicted_fluxes.min())
        max_flux = max(real_fluxes.max(), predicted_fluxes.max())
        ax.plot([min_flux, max_flux], [min_flux, max_flux], color='red', linestyle='--', label='1:1 Line')
        ax.legend()

        # Set axis limits
        min_corner = min(xedges[0], yedges[0])
        max_corner = max(xedges[-1], yedges[-1])
        if x_y_lim is not None:
            ax.set_xlim(min_corner, x_y_lim)
            ax.set_ylim(min_corner, x_y_lim)
        else:
            ax.set_xlim(min_corner, max_corner)
            ax.set_ylim(min_corner, max_corner)

        # Set title
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        if save:
            filepath = os.path.join(data_folder, f"{histogram_filename}.png")
            plt.savefig(filepath)
            print_box(f"2D histogram '{histogram_filename}.png' saved in {data_folder}!")
        else:
            plt.show()
        plt.close()

    def _make_plot(
            self,
            sources: np.ndarray,
            targets: np.ndarray,
            outputs: np.ndarray,
            predicted_psfs: np.ndarray,
            psfs: np.ndarray,
            titles: list[str],
            count: int,
            filename: str,
            data_folder: str,
            show_real_min_infered: bool = False,
            save: bool = False
        ) -> None:
        """
        Create a plot with the sources, targets, outputs, predicted PSFs, and PSFs.
        
        :param sources: Array of source images.
        :type sources: np.ndarray
        :param targets: Array of target images.
        :type targets: np.ndarray
        :param outputs: Array of cleaned images.
        :type outputs: np.ndarray
        :param predicted_psfs: Array of predicted PSF images.
        :type predicted_psfs: np.ndarray
        :param psfs: Array of PSF images.
        :type psfs: np.ndarray
        :param titles: List of titles for each model.
        :type titles: list[str]
        :param count: Count of the current image in the batch.
        :type count: int
        :param filename: Filename for saving the plot.
        :type filename: str
        :param data_folder: Folder to save the plot.
        :type data_folder: str
        :param show_real_min_infered: Whether to show the real minus inferred image.
        :type show_real_min_infered: bool
        :param save: Whether to save the plot or show it.
        :type save: bool
        :raises ValueError: If the input arrays do not have the same shape.
        """
        rows = sources.shape[0] # number of models

        cols = 5
        if show_real_min_infered:
            cols += 1

        interval = PercentileInterval(99.5)
            
        fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        for row in range(rows):
            source = sources[row]
            target = targets[row]
            output = outputs[row]
            diff_predicted = predicted_psfs[row]
            psf = psfs[row]

            norm_target = ImageNormalize(target, interval=interval, stretch=AsinhStretch())
            norm_psf = ImageNormalize(psf, interval=interval, stretch=AsinhStretch())
            norm_diff = ImageNormalize(diff_predicted, interval=interval, stretch=AsinhStretch())

            # Store the axes for the current row in variables
            if rows == 1:
                input_ax = ax[0]
                target_ax = ax[1]
                cleaned_ax = ax[2]
                diff_ax = ax[3]
                psf_ax = ax[4]
                if show_real_min_infered:
                    tar_min_out_ax = ax[5]
            else:
                input_ax = ax[row, 0]
                target_ax = ax[row, 1]
                cleaned_ax = ax[row, 2]
                diff_ax = ax[row, 3]
                psf_ax = ax[row, 4]
                if show_real_min_infered:
                    tar_min_out_ax = ax[row, 5]

            # Add a title for the row
            if rows > 1:
                ax[row, 0].annotate(
                    titles[row],
                    xy=(0, 0.5),
                    xytext=(-ax[row, 0].yaxis.labelpad - 5, 0),
                    xycoords=ax[row, 0].yaxis.label,
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90
                )

            # Plot the images
            _ = input_ax.imshow(source, norm=norm_target, cmap="gray")
            input_ax.set_title("Input")
            input_ax.axis("off")

            _ = target_ax.imshow(target, norm=norm_target, cmap="gray")
            target_ax.set_title("Target")
            target_ax.axis("off")

            _ = cleaned_ax.imshow(output, norm=norm_target, cmap="gray")
            cleaned_ax.set_title("Output")
            cleaned_ax.axis("off")

            _ = diff_ax.imshow(diff_predicted, norm=norm_diff, cmap="magma")
            diff_ax.set_title("Input - Output")
            diff_ax.axis("off")

            _ = psf_ax.imshow(psf, norm=norm_psf, cmap="magma")
            psf_ax.set_title("Input - Target (PSF)")
            psf_ax.axis("off")

            if show_real_min_infered:
                tar_min_out = target-output
                tar_min_out_ax.imshow(tar_min_out, cmap="magma")
                tar_min_out_ax.set_title("Target - Output")
                tar_min_out_ax.axis("off")

                # Create a colorbar for the last image
                cbar = fig.colorbar(tar_min_out_ax.images[0], ax=tar_min_out_ax, shrink=0.5, aspect=5)

                max_val = np.max(tar_min_out)
                min_val = np.min(tar_min_out)

                # Add a text under the last image
                tar_min_out_ax.text(
                    0.5,
                    -0.3,
                    f"Max: {max_val:.2f}\nMin: {min_val:.2f}",
                    ha='center',
                    va='center',
                    fontsize=12,
                    fontweight='bold',
                    transform=tar_min_out_ax.transAxes
                )

        plt.tight_layout()
        if save:
            filepath = os.path.join(data_folder, f"{filename}_{count}.png")
            plt.savefig(filepath)
            print_box(f"Image '{filename}_{count}.png' saved successfully in {data_folder}!")
        else:
            plt.show()
        plt.close()

    def _compatibility_check(
            self,
            sources: np.ndarray,
            targets: np.ndarray,
            outputs: np.ndarray,
            predicted_psfs: np.ndarray,
            psfs: np.ndarray
        ) -> None:
        """
        Check if all input arrays have the same shape and raise an error if not.
        """
        # Check if all arrays have the same shape
        if not (sources.shape == targets.shape == outputs.shape == predicted_psfs.shape == psfs.shape):
            raise ValueError("All input arrays must have the same number of images.")
        
    ######################################################################################################################################
    ##########################################           DEPRICATED                  #####################################################
    ######################################################################################################################################

    def make_3d_galaxy_plot(
            self,
            real_gal: list[np.ndarray],
            psfs: list[np.ndarray],
        ) -> None:
        real_gal = np.array(real_gal)  # (B, C, H, W)
        psfs = np.array(psfs)  # (B, C, H, W)
        validate_numpy_array(real_gal, ndim=4)
        validate_numpy_array(psfs, ndim=4)

        # Calculate the normalized values for the images
        real_gal = real_gal / np.max(real_gal, axis=(2, 3), keepdims=True) + 1e-6
        psfs = psfs / np.max(psfs, axis=(2, 3), keepdims=True) + 1e-6

        # Calculate the difference between the real galaxy and PSF
        diff = real_gal - psfs

        if real_gal.shape[0] != psfs.shape[0]:
            raise ValueError("The number of real galaxies and PSFs must be the same.")

        for i in range(real_gal.shape[0]):
            fig = plt.figure(figsize=(30, 5))

            norm = ImageNormalize(
                real_gal[i, 0],
                interval=PercentileInterval(99.5),
                stretch=AsinhStretch()
            )

            try:
                norm_psf = ImageNormalize(
                    psfs[i, 0],
                    interval=PercentileInterval(99.5),
                    stretch=AsinhStretch()
                )
            except IndexError:
                norm_psf = None

            # Print the index of the max value of the real galaxy and the PSF
            max_real_gal = np.unravel_index(np.argmax(real_gal[i, 0]), real_gal[i, 0].shape)
            max_psf = np.unravel_index(np.argmax(psfs[i, 0]), psfs[i, 0].shape)

            if max_psf != max_real_gal:
                # Shift the real galaxy and PSF so that the max of the PSF is at (64, 64)
                target_pos = max_real_gal
                shift_amount = (target_pos[0] - max_psf[0], target_pos[1] - max_psf[1])

                # Shift the arrays (preserving shape, using constant fill)
                psfs[i,0] = shift(psfs[i, 0], shift=shift_amount, order=1, mode='constant', cval=0.0)

            max_real_gal = np.unravel_index(np.argmax(real_gal[i, 0]), real_gal[i, 0].shape)
            max_psf = np.unravel_index(np.argmax(psfs[i, 0]), psfs[i, 0].shape)

            print_box(f"Max value of real galaxy {i}: {max_real_gal}, PSF: {max_psf}")

            # 1. Show real galaxy with imshow
            ax1 = fig.add_subplot(1, 5, 1)
            ax1.imshow(real_gal[i, 0], cmap='gray', origin='lower', norm=norm)
            ax1.set_title("Real Galaxy (imshow)")
            ax1.axis('off')

            # 2. 3D plot of real galaxy
            ax2 = fig.add_subplot(1, 5, 2, projection='3d')
            X = np.arange(real_gal[i, 0].shape[0])
            Y = np.arange(real_gal[i, 0].shape[1])
            X, Y = np.meshgrid(X, Y)
            ax2.plot_surface(X, Y, real_gal[i, 0], cmap='magma_r', edgecolor='none', norm=norm)
            ax2.set_title("Real Galaxy (3D)")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Intensity")

            # 3. 3D plot of PSF
            ax3 = fig.add_subplot(1, 5, 3, projection='3d')
            Xp = np.arange(psfs[i, 0].shape[0])
            Yp = np.arange(psfs[i, 0].shape[1])
            Xp, Yp = np.meshgrid(Xp, Yp)
            ax3.plot_surface(Xp, Yp, psfs[i, 0], cmap='magma_r', edgecolor='none', norm=norm_psf)
            ax3.set_title("PSF (3D)")
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_zlabel("Intensity")

            # 4. Show real galaxy - psf with imshow
            ax1 = fig.add_subplot(1, 5, 4)
            ax1.imshow(diff[i, 0], cmap='gray', origin='lower', norm=norm)
            ax1.set_title("Real Galaxy - PSF (imshow)")
            ax1.axis('off')

            # 5. 3D plot of real galaxy - psf
            ax4 = fig.add_subplot(1, 5, 5, projection='3d')
            Xd = np.arange(diff[i, 0].shape[0])
            Yd = np.arange(diff[i, 0].shape[1])
            Xd, Yd = np.meshgrid(Xd, Yd)
            ax4.plot_surface(Xd, Yd, diff[i, 0], cmap='magma_r', edgecolor='none', norm=norm)
            ax4.set_title("Real Galaxy - PSF (3D)")
            ax4.set_xlabel("X")
            ax4.set_ylabel("Y")
            ax4.set_zlabel("Intensity")

            plt.tight_layout()
            plt.savefig(f"galaxy_plot_{i}.png", bbox_inches='tight', dpi=200)
            print_box(f"3D galaxy plot for image {i} saved successfully as 'galaxy_plot_{i}.png'!")
            plt.close(fig)

    # Depricated:
    def make_frf_plot(
            self,
            flux_data_list: list[dict[str, np.ndarray]],
            model_names: list[str],
            flux_plot_datafolder: str,
            flux_plot_filename: str
        ) -> None:
        means_psf = []
        errors_psf = []

        means_gal = []
        errors_gal = []
        for flux_data in flux_data_list:
            # Extract keys and sort numerically
            sorted_keys = sorted(flux_data.keys(), key=lambda x: float(x.split('=')[1].strip()))
            # Remove 'f_agn = 0.5' (duplicate)
            if "f_agn = 0.5" in sorted_keys:
                sorted_keys.remove("f_agn = 0.5")

            x = []

            psf_means = []
            psf_errors = []

            gal_means = []
            gal_errors = []

            # Compute mean and error of frf
            for key in sorted_keys:
                f_val = f"{int(float(key.split('=')[1].strip()) * 100)}%"

                real_fluxes_psf = flux_data[key]["real_psf_fluxes"]
                predicted_fluxes_psf = flux_data[key]["predicted_psf_fluxes"]
                real_fluxes_gal = flux_data[key]["real_gal_fluxes"]
                predicted_fluxes_gal = flux_data[key]["predicted_gal_fluxes"]
                
                mean_frf_gal, sem_frf_gal = self._compute_frf_stats(real_fluxes_gal, predicted_fluxes_gal)
                mean_frf_psf, sem_frf_psf = self._compute_frf_stats(real_fluxes_psf, predicted_fluxes_psf)

                x.append(f_val)
                psf_means.append(mean_frf_psf)
                psf_errors.append(sem_frf_psf)
                gal_means.append(mean_frf_gal)
                gal_errors.append(sem_frf_gal)

            # Append the means and errors for the current model
            means_psf.append(psf_means)
            errors_psf.append(psf_errors)
            means_gal.append(gal_means)
            errors_gal.append(gal_errors)

        self._frf_plot(
            x=x,
            means=means_psf,
            errors=errors_psf,
            model_names=model_names,
            title=f"FRF (PSF) - {model_names[0]}" if len(model_names) == 1 else "FRF (PSF)",
            filepath=os.path.join(flux_plot_datafolder, f"{flux_plot_filename}_psf.png"),
            psf=True
        )
        self._frf_plot(
            x=x,
            means=means_gal,
            errors=errors_gal,
            model_names=model_names,
            title=f"FRF - {model_names[0]}" if len(model_names) == 1 else "FRF (PSF)",
            filepath=os.path.join(flux_plot_datafolder, f"{flux_plot_filename}_gal.png"),
            psf=False
        )

    # Depricated:
    def _compute_frf_stats(self, real_fluxes: np.ndarray, predicted_fluxes: np.ndarray) -> tuple[float, float]:
        frf = predicted_fluxes / (real_fluxes + 1e-8)

        frf = frf[frf < 5]
        frf = frf[frf > -5]

        n = len(frf)
        mean_frf = np.mean(frf)
        std_frf = np.std(frf, ddof=1)
        sem_frf = std_frf / np.sqrt(n)
        
        return mean_frf, 1.96*sem_frf

    def plot_binned_flux_summary(self, summary: pd.DataFrame, datafolder: str, filename: str) -> None:
        mean = summary["mean"]
        std = summary["std"]
        count = summary["count"]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a bar plot with error bars
        ax.bar(mean.index, mean.values, yerr=std.values, capsize=5)
        ax.set_xlabel("Flux Bins (Quantiles)")
        ax.set_ylabel("Mean Absolute Error")
        ax.grid()

        plt.tight_layout()
        plt.savefig(os.path.join(datafolder, f"{filename}.png"))
        plt.close()

    def _frf_plot(
            self,
            x: list[float],
            means: list[list[float]],
            errors: list[list[float]],
            model_names: list[str],
            title: str,
            filepath: str,
            psf: bool = False
        ):
        # Plotting
        _, ax = plt.subplots()

        for i in range(len(means)):
            this_mean = means[i]
            this_error = errors[i]
            this_model_name = model_names[i]
            if psf:
                x_plot = x[1:]
                this_mean = this_mean[1:]
                this_error = this_error[1:]
            else:
                x_plot = x

            x_pos = range(len(x_plot))

            ax.errorbar(x_pos, this_mean, yerr=this_error, fmt='o', capsize=5, label=this_model_name)

        ax.set_xlabel('AGN fraction (%)')
        ax.set_ylabel('FRF')
        ax.set_title(title)

        # Set x-ticks and labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_plot, rotation=45)

        # Plot the line at y=0
        ax.axhline(1, color='red', linestyle='--', label='Ideal (FRF = 1)')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
    def plot_psf_3d(
            self,
            psfs: list[np.ndarray],
            infered_psfs: list[np.ndarray],
            titles: list[str],
            filename: str,
            show_real_min_infered: bool = False
        ):
        psfs_arr = np.array(psfs) # (num_models, num_images, width, height)
        infered_psfs_arr = np.array(infered_psfs) # (num_models, num_images, width, height)

        for i in range(psfs_arr.shape[1]):
            psf = psfs_arr[:, i, :, :] # (num_models, width, height)
            infered_psf = infered_psfs_arr[:, i, :, :] # (num_models, width, height)

            # Create the plot
            self._make_psf_3d_plot(psf, infered_psf, titles, i, filename, show_real_min_infered=show_real_min_infered)

    def plot_psf_hist(self, psfs: np.ndarray, infered_psfs: np.ndarray, titles: str, filename: str) -> None:
        """
        Plot the histogram of the PSF and the infered PSF.

        :param psf: PSF image.
        :type psf: np.ndarray
        :param infered_psf: Infered PSF image.
        :type infered_psf: np.ndarray
        :param filename: Filename for saving the plot.
        :type filename: str
        """
        psfs_arr = np.array(psfs) # (num_models, num_images, width, height)
        infered_psfs_arr = np.array(infered_psfs) # (num_models, num_images, width, height)

        for i in range(psfs_arr.shape[1]):
            psf = psfs_arr[:, i, :, :] # (num_models, width, height)
            infered_psf = infered_psfs_arr[:, i, :, :] # (num_models, width, height)

            # Create the plot
            self._make_psf_hist(psf, infered_psf, titles, i, filename)

    def _make_psf_hist(self, psf: np.ndarray, infered_psf: np.ndarray, titles: str, count: int, filename: str) -> None:
        rows = psf.shape[0]
        cols = 5

        _, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        for row in range(rows):
            psf_image = psf[row]
            infered_psf_image = infered_psf[row]

            # Create a meshgrid for the axes
            X = np.arange(psf_image.shape[0])
            Y = np.arange(psf_image.shape[1])
            X, Y = np.meshgrid(X, Y)

            # Normalize the PSF images
            interval = PercentileInterval(99.5)
            stretch = AsinhStretch()
            norm_psf = ImageNormalize(psf_image, interval=interval, stretch=stretch)

            # Store the axes for the current row in variables
            if rows == 1:
                psf_ax = ax[0]
                plot_ax = ax[1]
                plot_ax2 = ax[2]

            else:
                psf_ax = ax[row, 0]
                plot_ax = ax[row, 1]
                plot_ax2 = ax[row, 2]


            # Add a title for the row
            if rows > 1:
                ax[row, 0].annotate(
                    titles[row],
                    xy=(0, 0.5),
                    xytext=(-ax[row, 0].yaxis.labelpad - 5, 0),
                    xycoords=ax[row, 0].yaxis.label,
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90
                )

            # Normalize the PSF images
            interval = PercentileInterval(99.5)
            stretch = AsinhStretch()
            norm_psf = ImageNormalize(psf_image, interval=interval, stretch=stretch)
            norm_infered_psf = ImageNormalize(infered_psf_image, interval=interval, stretch=stretch)

            # Calculate the range for the histogram based on the smallest and largest values
            min_val = min(norm_psf(psf_image).min(), norm_psf(infered_psf_image).min())
            max_val = max(norm_psf(psf_image).max(), norm_psf(infered_psf_image).max())

            # Plot the histograms with the specified range and make them fully transparent
            _ = psf_ax.hist(
                norm_psf(psf_image).flatten(),
                bins=10,
                range=(min_val, max_val),
                histtype='step',
                color='blue',
                edgecolor='blue',
            )
            _ = psf_ax.hist(
                norm_psf(infered_psf_image).flatten(),
                bins=10,
                range=(min_val, max_val),
                histtype='step',
                color='red',
                edgecolor='red'
            )
            psf_ax.set_title("PSF Histogram")
            psf_ax.legend(["PSF", "Infered PSF"])
            
            # Get the center of the PSF image
            center_x = psf_image.shape[0] // 2
            center_y = psf_image.shape[1] // 2

            # Get the data along the center of the PSF in x direction
            center_data = psf_image[center_x, :]
            center_infered_data = infered_psf_image[center_x, :]

            # Plot the data along the center of the PSF in y direction
            _ = plot_ax.plot(norm_psf(center_data), color='blue', label='PSF')
            _ = plot_ax.plot(norm_psf(center_infered_data), color='red', label='Infered PSF')
            plot_ax.set_title("PSF Profile along x")
            plot_ax.set_xlabel("Pixel")
            plot_ax.set_ylabel("Intensity")
            plot_ax.legend()

            # Get the data along the center of the PSF in y direction
            center_data = psf_image[:, center_y]
            center_infered_data = infered_psf_image[:, center_y]

            # Plot the data along the center of the PSF in y direction
            _ = plot_ax2.plot(norm_psf(center_data), color='blue', label='PSF')
            _ = plot_ax2.plot(norm_psf(center_infered_data), color='red', label='Infered PSF')
            plot_ax2.set_title("PSF Profile along y")
            plot_ax2.set_xlabel("Pixel")
            plot_ax2.set_ylabel("Intensity")
            plot_ax2.legend()
        plt.tight_layout()
        plt.savefig(f"{filename}_{count}.png")
        plt.close()
        

        info = f"Image '{filename}_{count}.png' saved successfully!"
        print_box(info)

    def _make_psf_3d_plot(
            self,
            psf: np.ndarray,
            infered_psf: np.ndarray,
            titles: list[str],
            count: int,
            filename: str,
            show_real_min_infered: bool = False
        ):
        rows = psf.shape[0]

        cols = 2
        if show_real_min_infered:
            cols += 1

        fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows), subplot_kw={"projection": "3d"})
        for row in range(rows):
            psf_image = psf[row]
            infered_psf_image = infered_psf[row]
    
            # Create a meshgrid for the axes
            X = np.arange(psf_image.shape[0])
            Y = np.arange(psf_image.shape[1])
            X, Y = np.meshgrid(X, Y)

            # Normalize the PSF images
            interval = PercentileInterval(99.5)
            stretch = AsinhStretch()
            norm_psf = ImageNormalize(psf_image, interval=interval, stretch=stretch)

            # Store the axes for the current row in variables
            if rows == 1:
                psf_ax = ax[0]
                infered_psf_ax = ax[1]
                if show_real_min_infered:
                    tar_min_out_ax = ax[2]
            else:
                psf_ax = ax[row, 0]
                infered_psf_ax = ax[row, 1]
                if show_real_min_infered:
                    tar_min_out_ax = ax[row, 2]

            # Add a title for the row
            if rows > 1:
                ax[row, 0].annotate(
                    titles[row],
                    xy=(0, 0.5),
                    xytext=(-ax[row, 0].yaxis.labelpad - 5, 0),
                    xycoords=ax[row, 0].yaxis.label,
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90
                )

            # Plot the images
            surf = psf_ax.plot_surface(X, Y, norm_psf(psf_image), cmap="magma", edgecolor="none")
            psf_ax.set_title("PSF")
            fig.colorbar(surf, ax=psf_ax, shrink=0.5, aspect=5)

            infered_surf = infered_psf_ax.plot_surface(X, Y, norm_psf(infered_psf_image), cmap="magma", edgecolor="none")
            infered_psf_ax.set_title("Infered PSF")
            fig.colorbar(infered_surf, ax=infered_psf_ax, shrink=0.5, aspect=5)

            if show_real_min_infered:
                tar_min_out = psf_image - infered_psf_image
                tar_min_out_surf = tar_min_out_ax.plot_surface(X, Y, norm_psf(tar_min_out), cmap="magma", edgecolor="none")
                tar_min_out_ax.set_title("PSF - Infered PSF")
                fig.colorbar(tar_min_out_surf, ax=tar_min_out_ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        plt.savefig(f"{filename}_{count}.png")
        plt.close()

        info = f"Image '{filename}_{count}.png' saved successfully!"
        print_box(info)