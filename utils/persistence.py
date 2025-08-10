import os
from copy import deepcopy
import pickle as pkl
import json
from typing import Any, Union
from pathlib import Path
import torch


def load_pkl_file(full_filepath: str) -> Any:
    """
    Load a pickle file from the specified path.

    :param full_filepath: The full path to the pickle file.
    :type full_filepath: str
    :return: The data loaded from the pickle file.
    :rtype: Any
    """
    if not full_filepath.endswith(".pkl"):
        full_filepath += ".pkl"
    
    # Get the DRAGN project directory (where this utils folder is located)
    dragn_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # If the path is relative, make it relative to DRAGN directory
    if not os.path.isabs(full_filepath):
        full_filepath = os.path.join(dragn_dir, full_filepath)
    
    if not os.path.exists(full_filepath):
        raise FileNotFoundError(f"File not found: {full_filepath}")
        
    with open(full_filepath, "rb") as file:
        data = pkl.load(file)
    return data

def save_pkl_file(data: Any, full_filepath: str):
    """
    Save data to a pickle file at the specified path.
		
    :param data: The data to be saved.
    :type data: Any
    :param full_filepath: The full path where the pickle file will be saved.
    :type full_filepath: str
    """
    full_filepath = str(full_filepath).replace(" ", "_")

    if not full_filepath.endswith(".pkl"):
        full_filepath += ".pkl"

    # Get the DRAGN project directory (where this utils folder is located)
    dragn_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # If the path is relative, make it relative to DRAGN directory
    if not os.path.isabs(full_filepath):
        full_filepath = os.path.join(dragn_dir, full_filepath)

    # Create the directory if it doesn't exist
    directory = os.path.dirname(full_filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(full_filepath, "wb") as file:
        pkl.dump(data, file)

    from utils.printing import print_box
    print_box(f"Data dumped in `{full_filepath}`")

def load_json_file(full_filepath: Union[str, Path]) -> Any:
    """
    Load a JSON file from the specified path.

    :param full_filepath: The full path to the JSON file.
    :type full_filepath: Union[str, Path]
    :return: The data loaded from the JSON file.
    :rtype: Any
    """
    full_filepath = str(full_filepath)
    if not full_filepath.endswith(".json"):
        full_filepath += ".json"

    # Get the DRAGN project directory (where this utils folder is located)
    dragn_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # If the path is relative, make it relative to DRAGN directory
    if not os.path.isabs(full_filepath):
        full_filepath = os.path.join(dragn_dir, full_filepath)

    if not os.path.exists(full_filepath):
        raise FileNotFoundError(f"File not found: {full_filepath}")

    with open(full_filepath, "r") as file:
        data = json.load(file)
    return data

def save_json_file(data: Any, full_filepath: Union[str, Path]):
    """
    Save data to a JSON file at the specified path.

    :param data: The data to be saved.
    :type data: Any
    :param full_filepath: The full path where the JSON file will be saved.
    :type full_filepath: Union[str, Path]
    """
    full_filepath = str(full_filepath).replace(" ", "_")

    if not full_filepath.endswith(".json"):
        full_filepath += ".json"

    # Create a copy of the data (used for serialization)
    data_cp = deepcopy(data)
    with open(full_filepath, "w") as file:
        json.dump(data_cp, file, indent=4)
