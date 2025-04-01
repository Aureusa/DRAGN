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
    for key, files in file_groups.items():
        if not re.search(pattern_agn_free, files[0]):
            # if the file does not match the pattern raise an error
            raise ValueError(
                f"Key: {key} does not match the pattern {pattern_agn_free}"
            )
        
        for file in files[1:]:
            source.append(file)
            target.append(files[0])

    print_box(f"Number of source-target pairs: {len(source)}")

    return source, target


def test_train_val_split(
        X : list[str],
        y : list[str],
        test_size=0.2,
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

    info = f"Training set size: {len(X_train)}\n"
    info += f"Validation set size: {len(X_val)}\n"
    info += f"Test set size: {len(X_test)}\n"
    info += f"Total dataset size: {len(X)}"
    print_box(info)
    
    return X_train, X_val, X_test, y_train, y_val, y_test