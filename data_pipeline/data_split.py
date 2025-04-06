import re
from sklearn.model_selection import train_test_split

from data_pipeline.getter import TELESCOPES_DB
from utils import print_box


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
    pattern_agn_free = TELESCOPES_DB["AGN_FREE_PATERN"]

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