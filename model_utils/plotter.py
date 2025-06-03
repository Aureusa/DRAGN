import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import re
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

from model_utils.performance_analysis import PAdict
from loggers_utils import log_execution
from utils import print_box


class Plotter:
    def plot_loss(self, train_loss: list[float], val_loss: list[float], best_val_loss: list[str], model_name: str, filepath: str) -> None:
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
        plt.plot(val_loss, label=f"Validation Loss; Best: {best_val_loss:.4f}")
        plt.title("Train and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(filepath ,f"{model_name}_loss.png"))

        print_box(f"Loss plot for '{model_name}' saved successfully!")

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

    def plot_cleaned_images_report(
            self,
            sources: list[np.ndarray],
            targets,
            outputs,
            titles,
            filename,
            f_agn
        ) -> None:
        """
        Create a tight grid of images for report/figure, similar to pix2pix paper.
        Columns: Input | Target | Output (per model)
        Rows: Each image in the batch.
        """
        # Convert lists to numpy arrays
        sources_arr = np.array(sources)  # (num_models, num_images, width, height)
        targets_arr = np.array(targets)  # (num_models, num_images, width, height)
        outputs_arr = np.array(outputs)  # (num_models, num_images, width, height)

        # Transpose arrays to shape (num_images, ...) for easier iteration
        # sources_arr: (num_models, num_images, w, h) -> (num_images, num_models, w, h)
        sources_arr = np.transpose(sources_arr, (1, 0, 2, 3))
        targets_arr = np.transpose(targets_arr, (1, 0, 2, 3))
        outputs_arr = np.transpose(outputs_arr, (1, 0, 2, 3))

        num_images = sources_arr.shape[0]
        num_models = outputs_arr.shape[1]

        # Prepare normalization (use target normalization for all)
        interval = PercentileInterval(99.5)
        norms = [ImageNormalize(targets_arr[i, 0], interval=interval, stretch=AsinhStretch()) for i in range(num_images)]

        # Figure: rows = num_images, cols = 2 + num_models
        cols = 2 + num_models
        ax = plt.subplots(num_images, cols, figsize=(3 * cols, 3 * num_images))[1]

        # If only one image, ax is 1D
        if num_images == 1:
            ax = np.expand_dims(ax, 0)
        if cols == 1:
            ax = np.expand_dims(ax, 1)

        for i in range(num_images):
            # Input (first model's input for each image)
            ax[i, 0].imshow(sources_arr[i, 0], norm=norms[i], cmap="gray")
            ax[i, 0].axis("off")
            # Add f_agn text to the top left
            ax[i, 0].text(
                0.02, 0.08, f"f_AGN = 0.{f_agn[i]}",
                color="white", fontsize=12, fontweight="bold",
                ha="left", va="top", transform=ax[i, 0].transAxes,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
            )
            # Target (first model's target for each image)
            ax[i, 1].imshow(targets_arr[i, 0], norm=norms[i], cmap="gray")
            ax[i, 1].axis("off")
            # Outputs for each model
            for m in range(num_models):
                ax[i, 2 + m].imshow(outputs_arr[i, m], norm=norms[i], cmap="gray")
                ax[i, 2 + m].axis("off")

        # Set column titles
        col_titles = ["Input", "Target"] + list(titles)
        for j, title in enumerate(col_titles):
            ax[0, j].set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(f"{filename}_report.png", bbox_inches="tight", dpi=200)
        plt.close()
        print_box(f"Report grid plot '{filename}_report.png' saved successfully!")


    @log_execution("Making plots...", "Plots made successfully!")
    def plot_cleaned_images(
            self,
            sources: list[np.ndarray],
            targets,
            outputs,
            diffs,
            psfs,
            titles,
            filename,
            show_real_min_infered: bool = False,
        ) -> None:
        """
        Plot the cleaned images, sources, targets, and PSFs.

        :param sources: List of source images.
        :type sources: list[np.ndarray]
        :param targets: List of target images.
        :type targets: list[np.ndarray]
        :param outputs: List of cleaned images.
        :type outputs: list[np.ndarray]
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
        outputs_arr = np.array(outputs)
        diffs_arr = np.array(diffs)
        psfs_arr = np.array(psfs)

        # Check if all arrays have the same shape
        self._compatibility_check(sources_arr, targets_arr, outputs_arr, diffs_arr, psfs_arr)

        for i in range(sources_arr.shape[1]):
            # Get the images for the batch of images
            sources = sources_arr[:, i, :, :] # (num_models, width, height)
            targets = targets_arr[:, i, :, :]
            outputs = outputs_arr[:, i, :, :]
            diffs = diffs_arr[:, i, :, :]
            psfs = psfs_arr[:, i, :, :]

            # Create the plot
            self._make_plot(sources, targets, outputs, diffs, psfs, titles, i, filename, show_real_min_infered=show_real_min_infered)

    def make_trend_plots(
            self,
            data_list: list[PAdict],
            filepath: str
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

                # Make the plot more readable if agn_frac is too large
                if len(agn_fracs) > 25:
                    frac = len(agn_fracs) // 25
                    agn_fracs = agn_fracs[::frac]
                    metric_values = metric_values[::frac]
                    agn_fracs = [f"{int(float(f.split(' = ')[1].split(' ')[0]) * 100)}%" for f in agn_fracs]  # Convert to percentage
                else:
                    agn_fracs = [f"{int(float(f.split(' = ')[1].split(' ')[0]) * 100)}%" for f in agn_fracs]

                plt.plot(agn_fracs, metric_values, marker='o', label=model_names[i])

            y_label = metric
            if re.search(r"PSNR", metric, re.IGNORECASE):
                y_label = metric + " (dB)"
            if re.search(r"Centroid", metric, re.IGNORECASE):
                y_label = metric + " (pixels)"
            if re.search(r"FRF", metric, re.IGNORECASE):
                plt.axhline(y=0, color='red', linestyle='--', label='Perfect FRF')

            plt.xlabel("AGN Fraction (f_agn - %)")
            plt.ylabel(y_label)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(filepath, f"{metric.lower().replace(' ', '_')}_trend_plot.png"))
            plt.close()

            info = f"Trend plot for '{metric}' saved successfully!"
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

    def _make_plot(
            self,
            sources: np.ndarray,
            targets: np.ndarray,
            outputs: np.ndarray,
            diffs: np.ndarray,
            psfs: np.ndarray,
            titles: list[str],
            count: int,
            filename: str,
            show_real_min_infered: bool = False
        ) -> None:
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
            diff_predicted = diffs[row]
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
        plt.savefig(f"{filename}_{count}.png")
        plt.close()

        info = f"Image '{filename}_{count}.png' saved successfully!"
        print_box(info)


    def make_2d_histogram(
            self,
            real_fluxes: np.ndarray,
            predicted_fluxes: np.ndarray,
            histogram_filename: str,
            histogram_datafolder: str,
            x_label: str = "Real PSF Flux",
            y_label: str = "Predicted PSF Flux",
            title: str = "2D Histogram of Real vs Predicted PSF Fluxes",
            gal: bool = False
        ) -> None:
        info = "Creating 2D histogram..."
        print_box(info)
        
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a copy of the plasma colormap and set the lowest value to white
        plasma = cm.get_cmap('plasma', 256)
        newcolors = plasma(np.linspace(0, 1, 256))
        newcolors[0, :] = np.array([1, 1, 1, 1])  # RGBA for white
        white_plasma = mpl.colors.ListedColormap(newcolors)

        hist, xedges, yedges = np.histogram2d(real_fluxes, predicted_fluxes, bins=35)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Apply logarithmic scaling to the histogram
        hist = np.log1p(hist)  # Use log1p to avoid log(0)

        im = ax.imshow(hist.T, extent=extent, origin='lower', aspect='auto', cmap=white_plasma)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Logarithmic Count")

        # Set axis limits to match the histogram
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])

        # Add a 1:1 line
        min_flux = min(real_fluxes.min(), predicted_fluxes.min())
        max_flux = max(real_fluxes.max(), predicted_fluxes.max())
        ax.plot([min_flux, max_flux], [min_flux, max_flux], color='red', linestyle='--', label='1:1 Line')
        ax.legend()

        filepath = os.path.join(histogram_datafolder, f"{histogram_filename}.png")

        # Stretch axes so the 1:1 line goes from bottom left to top left (vertical)
        min_corner = min(xedges[0], yedges[0])
        max_corner = max(xedges[-1], yedges[-1])
        if gal:
            ax.set_xlim(min_corner, 40000)
            ax.set_ylim(min_corner, 40000)
        else:
            ax.set_xlim(min_corner, max_corner)
            ax.set_ylim(min_corner, max_corner)

        ax.set_aspect('equal', adjustable='box')  # Allow axes to stretch vertically

        # # Plot a vertical 1:1 line (x = min_corner)
        # ax.plot([min_corner, min_corner], [min_corner, max_corner], color='red', linestyle='--', label='1:1 Line')
        # ax.legend()

        plt.savefig(filepath)
        plt.close()

        info = f"2D histogram '{histogram_filename}.png' saved successfully!"
        print_box(info)

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

    def _compatibility_check(self, sources: np.ndarray, targets, outputs, diffs, psfs) -> None:
        # Check if all arrays have the same shape
        if not (sources.shape[0] == targets.shape[0] == outputs.shape[0] == diffs.shape[0] == psfs.shape[0]):
            raise ValueError("All input arrays must have the same number of images.")