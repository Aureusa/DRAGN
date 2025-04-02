import os
import json
import glob
import re
from collections import defaultdict

from utils import print_box


# Define the scratch dir to use in habrok
SCRATCH_DIR = "/scratch"

# Load the data database
db_path = os.path.join(os.path.join(os.getcwd(), "data_pipeline"), "data_db.json")
with open(db_path, "r") as f:
    TELESCOPES_DB = json.load(f)


class FilepathGetter:
    def __init__(self, telescope: str, redshift: list[str]|None = None, redshift_treshhold: float|None = None) -> None:
        """
        Initialize the DataGetter class.
        This class is responsible for getting the filenames 
        from the telescope source path and grouping them by (snXXX, unique_id).

        :param telescope: The name of the telescope to get data from.
        :type telescope: str
        :param redshift: The list of redshift values to filter the data.
        :type redshift: list[str], optional
        :param redshift_treshhold: The redshift threshold to filter the data.
        Retrieves all files with redshift smaller than the threshold.
        Takes priority over `redshift`.
        :type redshift_treshhold: float, optional
        :raises ValueError: If the telescope is not supported or if the redshift
        is not supported for the given telescope.
        """
        if redshift_treshhold is not None:
            if redshift is not None:
                redshift = None
                info = "`redshift_treshhold` takes priority over `redshift`!\n"
                info += "All files bellow the treshhold value will be collected."
                print_box(info)
        if telescope not in TELESCOPES_DB:
            raise ValueError(f"Telescope {telescope} not supported. Choose from {list(TELESCOPES_DB.keys())}.")
        
        # Get the data folder from the database
        data_folder = TELESCOPES_DB[telescope]["folder"]

        # Get the path to root folder
        os.chdir("..")
        os.chdir("..")

        # Get the path to the telescope source path
        telecope_source_path = os.path.join(SCRATCH_DIR, "s4683099", data_folder)

        if redshift is not None:
            info = self._retrieve_files_from_snapnum_list(telescope, redshift, telecope_source_path)
        elif redshift_treshhold is not None:
            info = self._retrieve_files_with_z_treshhold(telescope, redshift_treshhold, telecope_source_path)
        else: # Collect all the files from the telescope source path
            info = "Retrieving the observations for all redshift values:\n"

            self.fits_files = glob.glob(f"{telecope_source_path}/**/*.fits", recursive=True)

            info += f"Found {len(self.fits_files)} .fits files in {telecope_source_path}"

        print_box(info)

    def _retrieve_files_with_z_treshhold(self, telescope: str, redshift_treshhold: float, telecope_source_path: str) -> str:
        """
        Retrieve the files from the telescope source path with redshift smaller than the threshold.
        
        :param telescope: The name of the telescope to get data from.
        :type telescope: str
        :param redshift_treshhold: The redshift threshold to filter the data.
        :type redshift_treshhold: float
        :param telecope_source_path: The path to the telescope source folder.
        :type telecope_source_path: str
        :return: A string with the information about the retrieved files.
        :rtype: str
        """
        info = f"Retrieving the observations for redshift smaller than <{redshift_treshhold}:\n"
        info += "snapnum folder - correspodning redshift\n"

        files = []

        telescope_avaliable_redshifts = set(TELESCOPES_DB[telescope]["redshifts"])
        for snapnum, z in TELESCOPES_DB["SNAP-REDSHIFT MAP"].items():
            if snapnum in telescope_avaliable_redshifts and z < redshift_treshhold:
                snapnum_folders = f"snapnum_{snapnum}"
                info += f"{snapnum_folders} - {round(z,2)}\n"

                # Get the path for the current redshift
                telecope_source_path_redshift = os.path.join(telecope_source_path, snapnum_folders)

                # Get all .fits files at the current redshift
                files_at_curr_redshift = glob.glob(os.path.join(telecope_source_path_redshift, "*.fits"))

                # Add the files to the list
                files.extend(files_at_curr_redshift)
        
        self.fits_files = files

        info += f"Found {len(self.fits_files)} .fits files for redshift smaller than <{redshift_treshhold}."

        return info
    
    def _retrieve_files_from_snapnum_list(self, telescope: str, redshift: list[str], telecope_source_path: str) -> str:
        """
        Retrieve the files from the telescope source path for the given redshift values.
        The redshift values are expected to be in the format: XXX.

        :param telescope: The name of the telescope to get data from.
        :type telescope: str
        :param redshift: The list of redshift values to filter the data.
        :type redshift: list[str]
        :param telecope_source_path: The path to the telescope source folder.
        :type telecope_source_path: str
        :raises ValueError: If the provided redshift value is not in the
        available redshifts for the telescope.
        :return: A string with the information about the retrieved files.
        :rtype: str
        """
        # Create a set of available redshifts for the telescope
        avaliable_redshifts = set(TELESCOPES_DB[telescope]["redshifts"])

        info = "Retrieving the observations for the following redshift values:\n"
        info += "snapnum folder - correspodning redshift\n"

        files = []
        
        # Loop through the redshift values
        for redshift_value in redshift:
            # Check if the provided redshift value is in the available redshifts
            if redshift_value not in avaliable_redshifts:
                raise ValueError(f"Data for redshift '{redshift}' not supported for telescope '{telescope}'.")
            info += f"snapnum_{redshift_value} - {round(TELESCOPES_DB['SNAP-REDSHIFT MAP'][redshift_value],2)}\n"
            # Get the path for the current redshift
            telecope_source_path_redshift = os.path.join(telecope_source_path, f"snapnum_{redshift_value}")

            # Get all .fits files at the current redshift
            files_at_curr_redshift = glob.glob(os.path.join(telecope_source_path_redshift, "*.fits"))

            # Add the files to the list
            files.extend(files_at_curr_redshift)

        self.fits_files = files

        info += f"Found {len(self.fits_files)} .fits files in {telecope_source_path} for the given redshifts."

        return info

    def get_data(self) -> tuple[dict, list]:
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
        pattern_agn = re.compile(rf"{TELESCOPES_DB['AGN_CONTAMINATION_PATTERN']}")
        pattern_agn_free = re.compile(rf"{TELESCOPES_DB['AGN_FREE_PATERN']}")
        
        # Set to store unique keys
        all_keys = set()

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

        print_box(f"Found {len(file_groups)} unique (snXXX, unique_id) pairs.")

        return file_groups, list(all_keys)
    