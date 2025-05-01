import torch
from torch.utils.data import DataLoader
from collections import Counter
import pickle
import os
from utils import print_box
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from astropy.visualization import AsinhStretch, ImageNormalize

from data_pipeline import GalaxyDataset
from data_pipeline.getter import TELESCOPES_DB
from model import AVALAIBLE_MODELS
from model_utils.metrics import get_metrics
from model_utils.performance_analysis import PAdict
from utils import load_pkl_file, save_pkl_file


class Plotter:
    def plot_loss(self, train_loss: list[float], val_loss: list[float], model_name: str) -> None:
        """
        Plot the training and validation loss.

        :param train_loss: List of training loss values.
        :type train_loss: list[float]
        :param val_loss: List of validation loss values.
        :type val_loss: list[float]
        :param model_name: Name of the model. Used to save the plot.
        :type model_name: str
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{model_name}_loss.png")

    def plot_cleaned_images(
            self,
            sources: list[np.ndarray],
            targets,
            cleaned_images,
            diffs,
            psfs,
            titles,
            norms: list[ImageNormalize],
            filename
        ) -> None:
        """
        Plot the cleaned images, sources, targets, and PSFs.

        :param sources: List of source images.
        :type sources: list[np.ndarray]
        :param targets: List of target images.
        :type targets: list[np.ndarray]
        :param cleaned_images: List of cleaned images.
        :type cleaned_images: list[np.ndarray]
        :param diffs: List of difference images.
        :type diffs: list[np.ndarray]
        :param psfs: List of PSF images.
        :type psfs: list[np.ndarray]
        :param titles: List of titles for each model.
        :type titles: list[str]
        :param norms: List of normalization objects for each image.
        :type norms: list[ImageNormalize]
        :param filename: Filename for saving the plot.
        :type filename: str
        """
        # Convert lists to numpy arrays
        sources_arr = np.array(sources) # (num_models, num_images, width, height)
        targets_arr = np.array(targets)
        cleaned_images_arr = np.array(cleaned_images)
        diffs_arr = np.array(diffs)
        psfs_arr = np.array(psfs)

        # Check if all arrays have the same shape
        self._compatibility_check(sources_arr, targets_arr, cleaned_images_arr, diffs_arr, psfs_arr)

        for i in range(sources_arr.shape[1]):
            # Get the images for the batch of images
            sources = sources_arr[:, i, :, :] # (num_models, width, height)
            targets = targets_arr[:, i, :, :]
            cleaned_images = cleaned_images_arr[:, i, :, :]
            diffs = diffs_arr[:, i, :, :]
            psfs = psfs_arr[:, i, :, :]
            norm = norms[i] 

            # Create the plot
            self._make_plot(sources, targets, cleaned_images, diffs, psfs, titles, norm, i, filename)

    def make_trend_plots(
            self,
            data_list: list[PAdict]
        ) -> None:
        """
        Make trend plot for any number of metrics and any number of models. You
        can pass any number of performance analysis DataFrames and the function
        will create a trend plot for each metric in the DataFrames.
        The trend plot will show the trend of the metric with respect to the AGN
        fraction (f_agn) for each model.

        :param data_list: List of performance analysis dicts.
        :type data_list: list[PAdict]
        :raises IndexError: If the DataFrames have different index or columns.
        """
        # Sort the DataFrames by index (f_agn) and check
        # if all dataframes have the same rows and columns
        sorted_dfs = []
        model_names = []
        for i, pa in enumerate(data_list):
            # Perform sanity check to ensure all DataFrames have the same index and columns
            if pa != data_list[0]:
                raise IndexError("DataFrames have different index or columns. Please check the input data.")

            # Sort the DataFrame by index (f_agn)
            df = pa.get_df(sort_by_fagn=True)
            sorted_dfs.append(df)
            model_names.append(pa.model_name)

        # Get the rows (indecies) and columns (metrics) from the first DataFrame
        # they should be the same for all DataFrames       
        indecies = sorted_dfs[0].index
        metrics = sorted_dfs[0].columns

        # Create a figure for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))

            for i, df in enumerate(sorted_dfs): # Iterate through the sorted performance analysis DataFrames
                y = df[metric]
                
                # Find the index for the nan values
                nan_indices = np.where(np.isnan(y))[0]

                # Remove the nan values from the data
                metric_values = np.delete(y, nan_indices)
                agn_fracs = np.delete(indecies, nan_indices)

                plt.plot(agn_fracs, metric_values, marker='o', label=model_names[i])

            plt.title(f"{metric} Trend")
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{metric}_trend_plot.png")
            plt.close()

            info = f"Trend plot for '{metric}' saved successfully!"
            print_box(info)

    def _make_plot(self, sources: np.ndarray, targets, cleaned_images, diffs, psfs, titles, norm, count, filename) -> None:
        rows = sources.shape[0] # number of models

        fig, ax = plt.subplots(rows, 5, figsize=(15, 5 * rows))
        for row in range(rows):
            source = sources[row]
            target = targets[row]
            cleaned_image = cleaned_images[row]
            diff_predicted = diffs[row]
            psf = psfs[row]


            # Store the axes for the current row in variables
            if rows == 1:
                input_ax = ax[0]
                target_ax = ax[1]
                cleaned_ax = ax[2]
                diff_ax = ax[3]
                psf_ax = ax[4]
            else:
                input_ax = ax[row, 0]
                target_ax = ax[row, 1]
                cleaned_ax = ax[row, 2]
                diff_ax = ax[row, 3]
                psf_ax = ax[row, 4]

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
            _ = input_ax.imshow(source, norm=norm, cmap="gray")
            input_ax.set_title("Input Image")
            input_ax.axis("off")

            _ = target_ax.imshow(target, norm=norm, cmap="gray")
            target_ax.set_title("Target Image")
            target_ax.axis("off")

            _ = cleaned_ax.imshow(cleaned_image, norm=norm, cmap="gray")
            cleaned_ax.set_title("Cleaned Image")
            cleaned_ax.axis("off")

            _ = diff_ax.imshow(diff_predicted, norm=norm, cmap="gray")
            diff_ax.set_title("Difference Image")
            diff_ax.axis("off")

            im4 = psf_ax.imshow(psf, norm=norm, cmap="gray")
            psf_ax.set_title("PSF Image")
            psf_ax.axis("off")
            fig.colorbar(im4, ax=psf_ax)

        plt.tight_layout()
        plt.savefig(f"{filename}_{count}.png")
        plt.close()

        info = f"Image '{filename}_{count}.png' saved successfully!"
        print_box(info)


    def make_2d_histogram(self, real_psf_fluxes: np.ndarray, predicted_psf_fluxes: np.ndarray, histogram_filename: str) -> None:
        info = "Creating 2D histogram..."
        print_box(info)
        
        fig, ax = plt.subplots(figsize=(8, 6))

        hist, xedges, yedges = np.histogram2d(real_psf_fluxes, predicted_psf_fluxes, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Apply logarithmic scaling to the histogram
        hist = np.log1p(hist)  # Use log1p to avoid log(0)

        im = ax.imshow(hist.T, extent=extent, origin='lower', aspect='auto', cmap='plasma')
        ax.set_xlabel("Real PSF Flux")
        ax.set_ylabel("Predicted PSF Flux")
        ax.set_title("2D Histogram of Real vs Predicted PSF Fluxes (Logarithmic Scale)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Logarithmic Count")

        # Add a 1:1 line
        min_flux = min(real_psf_fluxes.min(), predicted_psf_fluxes.min())
        max_flux = max(real_psf_fluxes.max(), predicted_psf_fluxes.max())
        ax.plot([min_flux, max_flux], [min_flux, max_flux], color='red', linestyle='--', label='1:1 Line')
        ax.legend()

        plt.savefig(f"{histogram_filename}.png")
        plt.close()

        info = f"2D histogram '{histogram_filename}.png' saved successfully!"
        print_box(info)

    def make_flux_difference_plot(self, predicted_psf_fluxes: np.ndarray, real_psf_fluxes: np.ndarray, flux_plot_filename: str) -> None:
        info = "Creating flux difference plot..."
        print_box(info)
        # Create a plot for the difference between predicted and real PSF fluxes
        _, ax = plt.subplots(figsize=(8, 6))

        # Calculate the difference
        flux_difference = predicted_psf_fluxes - real_psf_fluxes

        # Bin the data
        # Bin the data
        bins = np.linspace(real_psf_fluxes.min(), real_psf_fluxes.max(), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        binned_differences = []

        for i in range(len(bins) - 1):
            # Get the slice of flux_difference for the current bin
            bin_slice = flux_difference[(real_psf_fluxes >= bins[i]) & (real_psf_fluxes < bins[i + 1])]
            
            # Check if the slice is empty
            if len(bin_slice) > 0:
                binned_differences.append(bin_slice.mean())
            else:
                binned_differences.append(np.nan)  # Use np.nan for empty bins

        # Plot the binned differences, ignoring NaN values
        valid_indices = ~np.isnan(binned_differences)
        ax.plot(bin_centers[valid_indices], np.array(binned_differences)[valid_indices], marker='o', label='Binned Difference', color='blue')

        # Add a horizontal line at 0 (perfect predictions)
        ax.axhline(0, color='red', linestyle='--', label='Perfect Prediction')

        ax.set_xlabel("Real PSF Flux")
        ax.set_ylabel("Difference (Predicted - Real)")
        ax.set_title("Difference Between Predicted and Real PSF Fluxes")
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{flux_plot_filename}.png")
        plt.close()

        info = f"Flux difference plot '{flux_plot_filename}.png' saved successfully!"
        print_box(info)


    def _compatibility_check(self, sources: np.ndarray, targets, cleaned_images, diffs, psfs) -> None:
        # Check if all arrays have the same shape
        if not (sources.shape[0] == targets.shape[0] == cleaned_images.shape[0] == diffs.shape[0] == psfs.shape[0]):
            raise ValueError("All input arrays must have the same number of images.")