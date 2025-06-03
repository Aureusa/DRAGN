"""
NOTE FOR USERS:

This module exposes the main astro_pipeline components for the DRAGN project,
including `StructuralParamsRegistry` and `Photometry`.
**This pipeline is not complete and is not fully integrated into the main DRAGN workflow.**

Example usage:
    data = ...  # Load your data as a tensor
    
    from astro_pipeline import Photometry
    phot = Photometry()
    registry = phot.perform_segmentation(data, max_workers=4, registry_name="example")

Refer to the documentation or code for more details.
"""
from astro_pipeline.param_registry import StructuralParamsRegistry
from astro_pipeline.photometry import Photometry
