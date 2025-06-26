from astropy.io import fits
from reproject import reproject_exact
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
from utils_utils.validation import validate_numpy_array
from scipy.ndimage import zoom


def load_fits_data(filepath: str, max_val=False) -> np.ndarray:
    """
    Load a FITS file and return the data
      as a numpy array.

    :param filepath: The path to the FITS file.
    :type filepath: str
    """
    try:
        with fits.open(filepath, memmap=True) as hdul:
            data = hdul[0].data

            if max_val:
                return np.nanmax(data)
        
        validate_numpy_array(data, ndim=(2,3))
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading FITS file {filepath}: {e}")
    
    
def center_crop(data, target_height, target_width, center_brightness=True):
    """
    Crop the image around the brightest pixel.

    :param data: 2D numpy array
    :param target_height: Height of the crop
    :param target_width: Width of the crop
    :return: Cropped 2D numpy array
    """
    def _adjust_top_left(
            top,
            left,
            bottom,
            right,
            target_height,
            target_width
        ):
        if bottom - top < target_height:
            top = max(bottom - target_height, 0)
        if right - left < target_width:
            left = max(right - target_width, 0)
        return top, left

    h, w = data.shape

    # Ensure the target dimensions are not larger than the image dimensions
    target_height = min(target_height, h)
    target_width = min(target_width, w)

    # Use the center of the image as the crop center
    center_y, center_x = h // 2, w // 2

    # Calculate crop boundaries
    top = max(center_y - target_height // 2, 0)
    left = max(center_x - target_width // 2, 0)
    bottom = min(top + target_height, h)
    right = min(left + target_width, w)

    # Adjust top/left if crop goes out of bounds
    top, left = _adjust_top_left(
        top,
        left,
        bottom,
        right,
        target_height,
        target_width
    )

    if center_brightness:
        # Crop the data to the calculated boundaries
        data_cropped = data[top:bottom, left:right]

        # Find the brightest pixel in the cropped region
        max_idx_cropped = np.unravel_index(np.argmax(data_cropped), data_cropped.shape)

        # Map the index back to the original data coordinates
        center_y = top + max_idx_cropped[0]
        center_x = left + max_idx_cropped[1]

        # Recalculate crop boundaries based on the brightest pixel in the original data
        top = max(center_y - target_height // 2, 0)
        left = max(center_x - target_width // 2, 0)
        bottom = min(top + target_height, h)
        right = min(left + target_width, w)

        # Adjust again if the crop goes out of bounds
        top, left = _adjust_top_left(
            top,
            left,
            bottom,
            right,
            target_height,
            target_width
        )

    return data[top:bottom, left:right]
