import random
import glob
import re
from collections import defaultdict

from utils import print_box


class ForgeData:
    """
    Class to handle the data splitting and file grouping for the AGN dataset.
    """

    def __init__(self, path: str):
        """
        Initialize the ForgeData class.

        :param path: The path to the folder containing the .fits files.
        :type path: str
        """
        self.path = path

    def forge_training_data(self): # MAIN FUNCTION
        """
        Main function to forge the training data.
        It retrieves the data, splits it into train, validation, and test sets,
        and creates source-target pairs.
        
        :return: A tuple containing the training, validation, and test sets.
        :rtype: tuple[list[str], list[str], list[str], list[str], list[str], list[str]]
        """
        # Get the data from the given path and group them by (snXXX, unique_id).
        file_groups, _ = self.get_data(self.path)

        # Split the data into training, validation, and test sets.
        train_dict, val_dict, test_dict = self.train_test_val_split(file_groups)
        
        # Create source-target pairs for training, validation, and test sets.
        X_train, y_train = self.create_source_target_pairs(train_dict)
        X_val, y_val = self.create_source_target_pairs(val_dict)
        X_test, y_test = self.create_source_target_pairs(test_dict)

        # Info
        print(f"Train: {len(X_train)}-{len(y_train)}")
        print(f"Validation: {len(X_val)}-{len(y_val)}")
        print(f"Test: {len(X_test)}-{len(y_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_data(self, path: str) -> tuple[dict, list]:
        """
        Get the data from the a given path and group them by (snXXX, unique_id).
        The files are grouped by the identifiers in the filename,
        which are expected to be in the format:
        snXXX_..._unique_id_... (for AGN Contamination)
        snXXX_..._unique_id.fits (for AGN Free)

        :param path: The path to the folder containing the .fits files.
        :type path: str
        :return: A tuple containing a dictionary with the grouped files and a list of unique keys.
        The dictionary keys are tuples of (snXXX, unique_id) and the values are lists of file paths.
        The list contains all unique (snXXX, unique_id) pairs found in the files.
        :rtype: tuple[dict, list]
        """
        fits_files = glob.glob(f"{path}/**/*.fits", recursive=True)

        # Dictionary to group files by (snXXX, unique_id)
        file_groups = defaultdict(list)

        # Regular expression to extract identifiers
        pattern_agn = re.compile(r"sn(\\d+)_.*?_(\\d+)_") # AGN Contamination pattern
        pattern_agn_free = re.compile(r"_sn(\\d+)_.*?_(\\d+).fits") # AGN Free pattern
        
        # Set to store unique keys
        all_keys = set()

        # Process each file
        for file in fits_files:
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
        
    def train_test_val_split(
            self,
            file_groups: dict,
            train_ratio: float = 0.7,
            val_ratio: float = 0.10,
            test_ratio: float = 0.20
        ) -> tuple[dict, dict, dict]:
        """
        Split the keys of a dictionary into train, validation, and test sets.

        :param file_groups: The dictionary to split. Keys are tuples, and values are lists of file paths.
        :type file_groups: dict
        :param train_ratio: Proportion of data to use for training.
        :type train_ratio: float
        :param val_ratio: Proportion of data to use for validation.
        :type val_ratio: float
        :param test_ratio: Proportion of data to use for testing.
        :type test_ratio: float
        :return: Three dictionaries: train_dict, val_dict, test_dict.
        :rtype: tuple[dict, dict, dict]
        """
        # Ensure the ratios sum to 1
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

        # Shuffle the keys to ensure randomness
        keys = list(file_groups.keys())
        random.shuffle(keys)

        # Compute split indices
        total_keys = len(keys)
        train_end = int(total_keys * train_ratio)
        val_end = train_end + int(total_keys * val_ratio)

        # Split the keys
        train_keys = keys[:train_end]
        val_keys = keys[train_end:val_end]
        test_keys = keys[val_end:]

        # Create train, validation, and test dictionaries
        train_dict = {key: file_groups[key] for key in train_keys}
        val_dict = {key: file_groups[key] for key in val_keys}
        test_dict = {key: file_groups[key] for key in test_keys}

        return train_dict, val_dict, test_dict

    def create_source_target_pairs(self, file_groups: dict) -> tuple[list[str], list[str]]:
        """
        Auxiliary function to create source-target pairs from the file groups.
        The source is the AGN image and the target is the AGN-free image.

        :param file_groups: A dictionary containing the file groups.
        :type file_groups: dict
        :raises ValueError: If the file does not match the pattern with having
        the first file as the AGN-free image.
        :return: A tuple containing the source and target lists.
        :rtype: tuple[list[str], list[str]]
        """
        pattern_agn_free = "_sn(\\d+)_.*?_(\\d+).fits"

        source = []
        target = []
        tot_targets_count = 0
        tot_sources_count = 0
        for _, files in file_groups.items():
            if any(re.search(pattern_agn_free, f) for f in files):
                for file in files:
                    if re.search(pattern_agn_free, file):
                        tot_targets_count += 1
                        for i in range(len(files)-1):
                            target.append(file)
                    else:
                        tot_sources_count += 1
                        source.append(file)

        info = f"Number of source-target pairs: {len(source)}-{len(target)}"
        info += f"\nTotal targets {tot_targets_count} (AGN free)"
        info += f"\nTotal sources {tot_sources_count} (AGN corrupted)"
        print(info)

        return source, target
    