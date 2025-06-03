import os
import pickle as pkl
from typing import Any


def print_box(message: str):
    """
    Print a message in a box format.
    The box is created using Unicode box-drawing characters.

    :param message: The message to be printed in the box.
    :type message: str
    """
    # Ensure the box starts on a new line
    print()

    #91
    lines = message.split('\n')
    for line in lines:
        if len(line) > 91:
            split_index = line.rfind(' ', 0, 91)
            if split_index != -1:
                lines.insert(lines.index(line) + 1, line[split_index + 1:])
                lines[lines.index(line)] = line[:split_index]
    max_length = 88
    border_up = '┌' + '─' * (max_length + 2) + '┐'
    border_down = '└' + '─' * (max_length + 2) + '┘'
    print(border_up)
    for line in lines:
        print(f'│ {line.ljust(max_length)} │')
    print(border_down)

def load_pkl_file(full_filepath: str) -> Any:
    """
    Load a pickle file from the specified path.

    :param full_filepath: The full path to the pickle file.
    :type full_filepath: str
    :return: The data loaded from the pickle file.
    :rtype: Any
    """
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
    full_filepath = full_filepath.replace(" ", "_")

    if not full_filepath.endswith(".pkl"):
        full_filepath += ".pkl"

    # Create the directory if it doesn't exist
    directory = os.path.dirname(full_filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(full_filepath, "wb") as file:
        pkl.dump(data, file)

    print_box(f"Data dumped in `{full_filepath}`")
