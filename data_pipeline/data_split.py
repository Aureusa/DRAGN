"""
NOTE FOR USERS:

The preferred way to obtain data splits for this package is to use the
`ForgeData` class and its `forge_training_data` method.

Example usage:
    from data_pipeline.data_split import ForgeData
    from data_pipeline.getter import FilepathGetter

    # Initialize the FilepathGetter with the desired telescope and redshift values.
    getter = FilepathGetter(path="path/to/data")
    forge = ForgeData()

    # Retrieve the file groups from the getter.
    file_groups, _ = getter.get_data()

    # Forge the training data using the ForgeData class.
    X_train, y_train, X_val, y_val, X_test, y_test = forge.forge_training_data(file_groups)

This ensures consistent and correct data preparation for all downstream tasks.
"""
import re
from sklearn.model_selection import train_test_split
import random

from data_pipeline._telescopes_db import TELESCOPES_DB
from loggers_utils import log_execution
from utils import print_box
from utils_utils.validation import validate_dict

# Derpicated function, use `ForgeData` class instead.
def create_source_target_pairs(file_groups: dict) -> tuple[list[str], list[str]]:
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
    pattern_agn_free = TELESCOPES_DB["AGN FREE PATTERN"]

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
    print_box(info)

    return source, target


# Derpicated function, use `ForgeData` class instead.
def test_train_val_split(
        X : list[str],
        y : list[str],
        test_size=0.1,
        val_size=0.1
    ) -> tuple:
    """
    Split the dataset into training, validation, and test sets.

    :param X: the features of the dataset
    :type X: list[str]
    :param y: the target of the dataset
    :type y: list[str]
    :param test_size: the test percentage, defaults to 0.2
    :type test_size: float, optional
    :param val_size: the validation percentage, defaults to 0.1
    :type val_size: float, optional
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    :rtype: tuple[list[float], list[float], list[float], list[float], list[float], list[float]]
    """
    # First split into train + temp and test
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=test_size
    )
    
    # Calculate the adjusted validation size as a proportion of the remaining data
    adjusted_val_size = val_size / (1 - test_size)
    
    # Now split the remaining data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=adjusted_val_size
    )

    info = f"Training set size: {len(X_train)} - {len(y_train)}\n"
    info += f"Validation set size: {len(X_val)} - {len(y_val)}\n"
    info += f"Test set size: {len(X_test)} - {len(X_test)}\n"
    info += f"Total dataset size: {len(X)} - {len(y)}"
    print_box(info)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class ForgeData:
    """
    Class to handle the data splitting and file grouping for the AGN dataset.
    """
    @log_execution("Forging data...", "Data forged successfully!")
    def forge_training_data(
            self,
            file_groups: dict,
            train_ratio: float = 0.7,
            val_ratio: float = 0.10,
            test_ratio: float = 0.20
        ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        """
        Main function to forge the training data.
        It retrieves the data, splits it into train, validation, and test sets,
        and creates source-target pairs.
        
        :param file_groups: The dictionary containing the file groups. The keys are tuples,
        and the values are lists of file paths contnaining at least 1 AGN free galaxy.
        This is expected to be the output of the `get_data` method of the `FilepathGetter` class.
        :type file_groups: dict
        :return: A tuple containing the training, validation, and test sets.
        :rtype: tuple[list[str], list[str], list[str], list[str], list[str], list[str]]
        """
        # Split the data into training, validation, and test sets.
        train_dict, val_dict, test_dict = self.train_test_val_split(
            file_groups,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # Create source-target pairs for training, validation, and test sets.
        X_train, y_train = self.create_source_target_pairs(train_dict)
        X_val, y_val = self.create_source_target_pairs(val_dict)
        X_test, y_test = self.create_source_target_pairs(test_dict)

        # Info
        info = f"Set: X-Y\n"
        info += f"Train: {len(X_train)}-{len(y_train)}\n"
        info += f"Validation: {len(X_val)}-{len(y_val)}\n"
        info += f"Test: {len(X_test)}-{len(y_test)}"
        print_box(info)

        return X_train, y_train, X_val, y_val, X_test, y_test
        
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
        validate_dict(file_groups, key_type=tuple, value_type=list)

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
        validate_dict(file_groups, key_type=tuple, value_type=list)
        
        pattern_agn_free = TELESCOPES_DB["AGN FREE PATTERN"]#"_sn(\\d+)_.*?_(\\d+).fits"

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
        print_box(info)

        return source, target
    