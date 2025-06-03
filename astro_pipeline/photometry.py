"""
NOTE FOR USERS:

This module provides a `Photometry` class for performing photometric
segmentation and extracting structural parameters from astronomical images.
**This module is not finished and is not used in the DRAGN project.**

- The implementation is experimental and may be incomplete or unstable.
- The main method demonstrates segmentation and parameter extraction
using `photutils`, but the class is not integrated into the main DRAGN pipeline.
- Most users do not need to use or modify this file for standard DRAGN workflows.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from photutils.background import Background2D
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog

from astro_pipeline.param_registry import StructuralParamsRegistry

class Photometry:
    """
    Class for performing photometry on astronomical images.
    This class provides methods to perform segmentation and extract
    structural parameters from images using photometric techniques.
    It uses the `photutils` library to detect sources and calculate
    structural parameters such as ellipticity, elongation, flux, and orientation.

    This class is not fully implemented yet, but it provides a
    `perform_segmentation` method that can be used to process images
    and extract structural parameters in parallel using a thread pool.
    """
    def perform_segmentation(self, data, max_workers, registry_name: str) -> StructuralParamsRegistry:
        """
        Perform segmentation on the input data and extract structural parameters
        such as ellipticity, elongation, flux, and orientation.
        
        :param data: The input data to process. It is expected to be a tensor
        containing images, where each image is a 2D array.
        :type data: torch.Tensor
        :param max_workers: The maximum number of worker threads to use for parallel processing.
        :type max_workers: int
        :param registry_name: The name of the registry to store the structural parameters.
        :type registry_name: str
        :return: A `StructuralParamsRegistry` containing the extracted structural parameters.
        :rtype: StructuralParamsRegistry
        """
        registry = StructuralParamsRegistry(registry_name)

        data = data.cpu().detach().numpy().squeeze(1)

        def process_image(im):
            # Remove negative values
            if np.any(im < 0):
                im = im - np.min(im) + 1e-6
            
            # Remove the background if possible
            try:
                # Create a 2D background model
                bkg = Background2D(im, (50, 50), exclude_percentile=20.00)

                background = bkg.background

                # Subtract the background from the image
                im_sub = im - background
            except ValueError:
                im_sub = im
                background = None


            # Detect sources in the image and create a segmentation map
            threshold = detect_threshold(im_sub, nsigma=3)
            segm = detect_sources(im_sub, threshold, npixels=5)

            # Check if the segmentation map is empty
            if segm is None:
                return None
            
            # Get the source in the middle
            ny, nx = segm.data.shape
            center = (ny // 2, nx // 2)
            center_label = segm.data[center]

            if center_label > 0:
                prop = SourceCatalog(im_sub, segm, background=background)
                segm.keep_label(center_label)
                segment = segm.make_source_mask()

                tab = {
                    "ellipticity": prop.ellipticity[0],
                    "elongation": prop.elongation[0],
                    "flux": im[segment].sum(),
                    "orientation": prop.orientation[0],
                }
            else:
                tab = None

            return tab

        # Parallelize the loop
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_image, data))
        
        # Add results to the registry in the same order as the data
        for result in results:
            registry.add(result)

        return registry