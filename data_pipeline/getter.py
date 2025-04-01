import os
import json
import glob
import re
from collections import defaultdict

from utils import print_box


# Define the scratch dir to use in habrok
SCRATCH_DIR = "scratch"

# Load the data database
db_path = os.path.join(os.path.join(os.getcwd(), "data_pipeline"), "data_db.json")
with open(db_path, "r") as f:
    TELESCOPES_DB = json.load(f)


class FilepathGetter:
    # def __init__(self, telescope, redshift=None) -> None:
    #     """
    #     Initialize the DataGetter class.
    #     This class is responsible for getting the filenames 
    #     from the telescope source path and grouping them by (snXXX, unique_id).

    #     :param telescope: The name of the telescope to get data from.
    #     :type telescope: str
    #     :param redshift: the redshift, defaults to None. If left to default,
    #     it will get all the filenames for the telescope.
    #     :type redshift: str, optional
    #     :raises ValueError: If the telescope is not supported or if the redshift
    #     is not supported for the given telescope.
    #     """
    #     if telescope not in TELESCOPES_DB:
    #         raise ValueError(f"Telescope {telescope} not supported. Choose from {list(TELESCOPES_DB.keys())}.")
        
    #     # Get the data folder from the database
    #     data_folder = TELESCOPES_DB[telescope]["folder"]

    #     telecope_source_path = os.path.join(os.path.expanduser("~"), SCRATCH_DIR, data_folder)

    #     # Validate the redshift
    #     if redshift is not None:
    #         if redshift not in TELESCOPES_DB[telescope]["redshifts"]:
    #             raise ValueError(f"Data for redshift '{redshift}' not supported for telescope '{telescope}'.")
            
    #         telecope_source_path = os.path.join(telecope_source_path, f"snapnum_{redshift}")

    #         # Get all .fits files
    #         self.fits_files = glob.glob(os.path.join(telecope_source_path, "*.fits"))
    #     else:
    #         self.fits_files = glob.glob(f"{telecope_source_path}/**/*.fits", recursive=True)

    #     print_box(f"Found {len(self.fits_files)} .fits files in {telecope_source_path}")

    def __init__(self):
        """
        USED FOR TESTING ONLY
        """
        # Define the folder path
        folder_path = "data"

        # Get all .fits files
        self.fits_files = glob.glob(os.path.join(folder_path, "*.fits"))

    def get_data(self, stop: int = 10) -> tuple[dict, list]:
        """
        Get the data from the telescope source path and group them by (snXXX, unique_id).
        The files are grouped by the identifiers in the filename, which are expected to be in the format:
        snXXX_..._unique_id_...
        snXXX_..._unique_id.fits

        :return: A tuple containing a dictionary with the grouped files and a list of unique keys.
        The dictionary keys are tuples of (snXXX, unique_id) and the values are lists of file paths.
        The list contains all unique (snXXX, unique_id) pairs found in the files.
        :rtype: tuple[dict, list]
        """
        # Dictionary to group files by (snXXX, unique_id)
        file_groups = defaultdict(list)

        # Regular expression to extract identifiers
        pattern_agn = re.compile(rf"{TELESCOPES_DB["AGN_CONTAMINATION_PATTERN"]}")
        pattern_agn_free = re.compile(rf"{TELESCOPES_DB["AGN_FREE_PATERN"]}")
        
        # Set to store unique keys
        all_keys = set()

        count = 0
        # Process each file
        for file in self.fits_files:
            match_agn = pattern_agn.search(file)
            match_agn_free = pattern_agn_free.search(file)
            if match_agn:
                sn_number = match_agn.group(1)  # e.g., snXXX
                unique_id = match_agn.group(2)  # e.g., unique_id
                key = (sn_number, unique_id)  # Create a tuple key
                all_keys.add(key)
                file_groups[key].append(file)
            if match_agn_free:
                sn_number = match_agn_free.group(1)
                unique_id = match_agn_free.group(2)
                key = (sn_number, unique_id)
                all_keys.add(key)
                file_groups[key].append(file)

            # # JUST FOR TESTING
            # count += 1
            # if count >= stop:
            #     break

        print_box(f"Found {len(file_groups)} unique (snXXX, unique_id) pairs.")

        return file_groups, list(all_keys)
    