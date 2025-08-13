import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (
    ImageNormalize,
    PercentileInterval,
    AsinhStretch,
    BaseStretch,
    BaseInterval,
)

from utils import print_box
from utils.validation import validate_numpy_array


def plot_loss(
        train_loss: list[float],
        val_loss: list[float],
        best_val_loss: float|None,
        filename: str,
        data_folder: str,
        xlabel: str = "Steps",
        ylabel: str = "Loss",
        title: str = "Train and Validation Loss"
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
    plt.plot(val_loss, label=f"Validation Loss; Best: {best_val_loss:.4f}" if best_val_loss else "Validation Loss")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(data_folder ,f"{filename}_loss.png"))

    print_box(f"Loss plot saved successfully in `{data_folder}`!")


def plot_image(self, img: np.ndarray, filename: str = "image_plot", data_folder: str = os.getcwd()) -> None:
    """
    Plot a single image.

    :param img: Image to plot. Should be a 2D numpy array.
    :type img: np.ndarray
    """
    plt.imshow(
        img,
        cmap='gray',
        norm=ImageNormalize(
            img,
            interval=PercentileInterval(99.5),
            stretch=AsinhStretch())
    )
    plt.axis('off')
    plt.savefig(os.path.join(data_folder, f"{filename}.png"))


class Plotter:
    def __init__(self, interval: BaseInterval = PercentileInterval(99.5), stretch: BaseStretch = AsinhStretch(), verbose: bool = False):
        self.interval = interval
        self.stretch = stretch
        self.verbose = verbose

    def _put_image_on_ax(self, ax: plt.Axes, img: np.ndarray, norm: Any|ImageNormalize = "standard", colorbar: bool = False, cmap: str = 'gray') -> None:
        im = ax.imshow(
            img,
            cmap=cmap,
            norm=self._norm(img) if norm == "standard" else norm
        )
        if colorbar:
            plt.colorbar(im, ax=ax)
        ax.axis('off')

    def _norm(self, img: np.ndarray) -> np.ndarray:
        return ImageNormalize(
            img,
            interval=self.interval,
            stretch=self.stretch
        )

    def _save(self, fig: plt.Figure, plotname:str, filepath: str) -> None:
        filepath = os.path.join(filepath, f"{plotname}.png")
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath, bbox_inches='tight', dpi=200)
        if self.verbose:
            print_box(f"Plot saved successfully as '{filepath}'!")

    def _reorder_to_known_pattern(self, f_agn, desired_pattern):
        """
        Reorder shuffled array to match a known repeating pattern.
        This function is absolute magic written by Claude, I do not
        understand it. We are lucky we don't use it for something
        extremely important ;)
        """
        from collections import defaultdict
    
        value_indices = defaultdict(list)
        for i, val in enumerate(f_agn):
            value_indices[val].append(i)
        
        # Calculate how many complete patterns we can make
        min_count = min(len(value_indices[val]) for val in desired_pattern)
        
        # Reorder using only the available complete patterns
        reorder_indices = []
        for rep in range(min_count):
            for val in desired_pattern:
                if rep < len(value_indices[val]):
                    reorder_indices.append(value_indices[val][rep])
        
        reordered = [f_agn[i] for i in reorder_indices]
        return reordered, reorder_indices

    def plot_square_grid(self, images: np.ndarray, filepath: str = os.getcwd(), save: bool = False) -> None:
        validate_numpy_array(images, ndim=4)

        # Set an upper limit for the square grid plot of 36 images
        allowed_sizes = [6,5,4,3,2,1]

        # Get the total number of images
        num_images = images.shape[0]

        # Find the size of the largest square grid
        for size in allowed_sizes:
            if size**2 <= num_images:
                break
        if size**2 < 4:
            raise ValueError(f"Not enough images to create a square grid of size {size}. Minimum is 4 images.")

        images = images[:size**2]  # Limit to the largest square

        fig, axes = plt.subplots(size, size, figsize=(15, 15))
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                self._put_image_on_ax(axes[i, j], images[idx])

        plt.tight_layout()
        if save:
            self._save(fig, f"square_grid_size_{size}", filepath)
        else:
            plt.show()
        plt.close()

    def plot_grid(
            self,
            sources: np.ndarray,
            targets: np.ndarray|None,
            outputs: np.ndarray,
            model_names: list[str],
            filename: str,
            data_folder: str,
            f_agn: list[int]|None,
            desired_pattern: list[int]|None,
            save: bool = False,
        ) -> None:
        """
        Create a tight grid of images for report/figure, similar to pix2pix paper.
        Columns: Input | Target | Output (per model)
        Rows: Each image in the batch.
        """
        total_num_images, _, _, _ = sources.shape

        if f_agn is not None and desired_pattern is not None and targets is not None:
            f_agn_reordered, reorder_indices = self._reorder_to_known_pattern(f_agn, desired_pattern)
            sources = sources[reorder_indices,:,:,:]
            targets = targets[reorder_indices,:,:,:]
            outputs = outputs[:,reorder_indices,:,:,:]
        else:
            f_agn_reordered = None

        # Use the len of the desired pattern for the rows (or 6 if not provided),
        max_imgs_per_fig = len(desired_pattern) if desired_pattern is not None else 6

        # Calculate how many complete figures and remaining images
        num_complete_figs = total_num_images // max_imgs_per_fig
        remaining_images = total_num_images % max_imgs_per_fig

        # Create complete figures first
        for fig_num in range(num_complete_figs):
            start_idx = fig_num * max_imgs_per_fig
            end_idx = start_idx + max_imgs_per_fig

            self._create_grid_figure(
                sources[start_idx:end_idx],
                targets[start_idx:end_idx] if targets is not None else None,
                outputs[:, start_idx:end_idx, :, :, :],
                f_agn=f_agn,
                f_agn_reordered=f_agn_reordered,
                model_names=model_names,
                filename=f"{filename}_{fig_num}",
                data_folder=data_folder,
                save=save
            )

        # Create final figure with remaining images if any
        if remaining_images > 0:
            start_idx = num_complete_figs * max_imgs_per_fig

            self._create_grid_figure(
                sources[start_idx:],
                targets[start_idx:] if targets is not None else None,
                outputs[:, start_idx:, :, :, :],
                f_agn=f_agn,
                f_agn_reordered=f_agn_reordered,
                model_names=model_names,
                filename=f"{filename}_{num_complete_figs}",
                data_folder=data_folder,
                save=save
            )

    def _create_grid_figure(self, sources: np.ndarray, targets: np.ndarray|None, outputs: np.ndarray, f_agn: list[float]|None, f_agn_reordered: list[float]|None, model_names: list[str], filename: str, data_folder: str, save: bool = False) -> None:
        """
        Create a grid figure for the sources, targets, and outputs.
        """
        num_images, _, _, _ = sources.shape
        num_models = outputs.shape[0] if len(outputs.shape) == 5 else 1

        # Use the number of images for the rows;
        # Use the number of models for the columns + 1 or 2 depending on if targets are available
        rows = num_images
        cols = 2 + num_models if targets is not None else 1 + num_models

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        for i in range(rows):
            for j in range(cols):
                ax = axes[i, j]
                if j == 0:
                    # Input (B, C, H, W)
                    self._put_image_on_ax(ax, sources[i, 0])
                    if f_agn is not None: # Add f_AGN = 0.** text to the top left of the input
                        ax.text(
                            0.02, 0.08, f"f_AGN = 0.{f_agn_reordered[i]}",
                            color="white", fontsize=12, fontweight="bold",
                            ha="left", va="top", transform=ax.transAxes,
                            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
                        )
                elif j == 1 and targets is not None:
                    # Target (B, C, H, W)
                    self._put_image_on_ax(ax, targets[i, 0])
                else:
                    # Output (NUM_MODELS, B, C, H, W)
                    self._put_image_on_ax(ax, outputs[j-2 if targets is not None else j-1, i, 0])

        # Set column titles
        if targets is not None:
            col_titles = ["Input", "Target"] + list(model_names)
        else:
            col_titles = ["Input"] + list(model_names)
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=14, fontweight="bold")

        fig.tight_layout()
        if save:
            self._save(fig, filename, data_folder)
        else:
            plt.show()

    def plot_diagnostic(
            self,
            sources: np.ndarray,
            targets: np.ndarray,
            outputs: np.ndarray,
            predicted_psfs: np.ndarray,
            psfs: np.ndarray,
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
        max_imgs_per_fig = 4
        num_images, _, _, _ = sources.shape
        
        # Calculate how many complete figures and remaining images
        num_complete_figs = num_images // max_imgs_per_fig
        remaining_images = num_images % max_imgs_per_fig
        
        # Create complete figures first
        for fig_num in range(num_complete_figs):
            start_idx = fig_num * max_imgs_per_fig
            end_idx = start_idx + max_imgs_per_fig
            rows = max_imgs_per_fig
            
            self._create_diagnostic_figure(
                sources[start_idx:end_idx],
                targets[start_idx:end_idx], 
                outputs[start_idx:end_idx],
                predicted_psfs[start_idx:end_idx],
                psfs[start_idx:end_idx],
                rows, fig_num, filename, data_folder, show_real_min_infered, save
            )
        
        # Create final figure with remaining images if any
        if remaining_images > 0:
            start_idx = num_complete_figs * max_imgs_per_fig
            rows = remaining_images  # Adjust rows to match remaining images
            
            self._create_diagnostic_figure(
                sources[start_idx:],
                targets[start_idx:],
                outputs[start_idx:], 
                predicted_psfs[start_idx:],
                psfs[start_idx:],
                rows, num_complete_figs, filename, data_folder, show_real_min_infered, save
            )

    def _create_diagnostic_figure(
            self,
            sources: np.ndarray,
            targets: np.ndarray,
            outputs: np.ndarray,
            predicted_psfs: np.ndarray,
            psfs: np.ndarray,
            rows: int,
            fig_num: int,
            filename: str,
            data_folder: str,
            show_real_min_infered: bool,
            save: bool
    ) -> None:
        # Convert (C,H,W) -> (H,W); assumes C=1
        sources = sources.squeeze()
        targets = targets.squeeze()
        outputs = outputs.squeeze()
        predicted_psfs = predicted_psfs.squeeze()
        psfs = psfs.squeeze()

        cols = 5
        if show_real_min_infered:
            cols += 1

        fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        for row in range(rows):
            input_ax = ax[row, 0]
            target_ax = ax[row, 1]
            output_ax = ax[row, 2]
            diff_ax = ax[row, 3]
            psf_ax = ax[row, 4]
            if show_real_min_infered:
                tar_min_out_ax = ax[row, 5]

            input_ax.set_title("Input")
            target_ax.set_title("Target")
            output_ax.set_title("Output")
            diff_ax.set_title("Input - Output")
            psf_ax.set_title("PSF (Input - Target)")
            if show_real_min_infered:
                tar_min_out_ax.set_title("Target - Output")

            self._put_image_on_ax(input_ax, sources[row])
            self._put_image_on_ax(target_ax, targets[row])
            self._put_image_on_ax(output_ax, outputs[row])
            self._put_image_on_ax(diff_ax, sources[row] - outputs[row])
            self._put_image_on_ax(psf_ax, psfs[row])
            if show_real_min_infered:
                self._put_image_on_ax(tar_min_out_ax, targets[row] - outputs[row], cmap="magma", colorbar=True, norm=None)

        plt.tight_layout()
        if save:
            self._save(fig, f"{filename}_{fig_num}", data_folder)
        else:
            plt.show()
        plt.close()
