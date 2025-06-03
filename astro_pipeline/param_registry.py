"""
NOTE FOR USERS:

This module provides the `StructuralParamsRegistry` class for registering,
managing, and merging structural parameters of galaxies.
It is designed to help organize and manipulate sets of structural parameters
(such as those derived from galaxy fitting using the `photutils` package)
in a consistent way.

**Key Points:**
- Use this class to add, merge, and extend collections of structural parameters,
each associated with a unique integer ID.
- The registry supports merging with other registries, concatenating parameter
dictionaries, and retrieving non-empty entries.
- All parameter sets should be dictionaries, and merging requires that both
registries have matching keys for each entry.
- This class is tailored for DRAGN project data structures and may require
adaptation for other use cases.

**Important:**
- Only use this class if your data follows the expected structure
(dictionary of parameters per object).
- For custom use, ensure your parameter dictionaries are compatible with
the registry's merging and concatenation logic.
- For more information, refer to the documentation or contact the maintainers.
"""
from typing import Any


class StructuralParamsRegistry:
    """
    A class to register and manage structural parameters for galaxies.
    The class allows for the addition, merging, and extension of structural
    parameters, as well as the retrieval of non-empty values and the
    concatenation of dictionaries.
    """
    def __init__(self, name: str) -> None:
        """
        Initialize the StructuralParamsRegistry object.
        
        :param name: The name of the registry.
        :type name: str
        """
        self.name = name

        # Make ordered set of integer ids starting from 0
        self._ids = set()

        # Make master dictionary to store the data
        # The keys are the ids and the values are the data
        self._data = {}

        # Initialize the number of none values and total values
        self._none_vals = 0
        self._total_vals = 0

        # Initialize the ids with none and valid values
        self._none_ids = set()
        self._valid_ids = set()

    @property
    def none_ids(self) -> set:
        """
        Get the ids of the None values in the registry.
        
        :return: The ids of the None values in the registry.
        :rtype: set
        """
        for k, v in self._data.items():
            if v is None:
                self._none_ids.add(k)
        return self._none_ids
    
    @property
    def valid_ids(self) -> set:
        """
        Get the ids of the valid values in the registry.
        
        :return: The ids of the valid values in the registry.
        :rtype: set
        """
        for k, v in self._data.items():
            if v is not None:
                self._valid_ids.add(k)
        return self._valid_ids

    @property
    def none_vals(self) -> int:
        """
        Get the number of None values in the registry.
        
        :return: The number of None values in the registry.
        :rtype: int
        """
        for k, v in self._data.items():
            if v is None:
                self._none_vals += 1
        return self._none_vals
    
    @property
    def total_vals(self) -> int:
        """
        Get the total number of values in the registry.
        
        :return: The total number of values in the registry.
        :rtype: int
        """
        for k, v in self._data.items():
            if v is not None:
                self._total_vals += 1
        return self._total_vals
    
    @property
    def ids(self) -> set:
        """
        Get the ids of the registry.
        
        :return: The ids of the registry.
        :rtype: set
        """
        return self._ids
    
    def __str__(self) -> str:
        """
        Return a string representation of the registry.
        """
        return f"Structural Parameters Registery `{self.name}`, {len(self._ids)} ids."

    def __getitem__(self, index: int) -> dict|None:
        """
        Get the data corresponding to the index.
        Indexing is done using the ids in the registry.
        
        :param index: The index of the data to get.
        :type index: int
        :return: The data corresponding to the index.
        :rtype: dict|None
        """
        # Check if the index is in the set
        if index not in self._ids:
            raise KeyError(
                f"Invalid index {index}. Maximum valid index is {max(self._ids) if self._ids else None}."
                )

        # Return the data corresponding to the index
        return self._data[index]
    
    def __len__(self) -> int:
        """
        Return the number of ids in the registry.
        
        :return: The number of ids in the registry.
        :rtype: int
        """
        return len(self._ids)

    def merge(self, other: Any) -> None:
        """
        Merge two StructuralParamsRegistry objects. The registries must have
        the same number of ids. Each valid entry in the both registries must have the same
        keys. The values of the same keys will be concatenated into a list.
        If both registries have None values, the value will be set to None.
        If one of the registries has a None value and the other registry has a
        valid value, the valid value will be used to fill the None value.
        If both registries have valid values, the values will be concatenated
        into a list.
        
        
        :param other: The other registry to merge with.
        :type other: StructuralParamsRegistry"""
        # Check if the other object is of the same type
        if not isinstance(other, StructuralParamsRegistry):
            raise TypeError(
                f"Expected a StructuralParamsRegistry object, got {type(other)}"
            )
        
        # Check if the two registries have the same number of ids
        if len(self) != len(other):
            raise ValueError(
                f"Cannot add two StructuralParamsRegistry objects of different lengths: {len(self)} and {len(other)}."
            )
        
        # Check if the valid values of each registries have the same keys
        keys = self._check_if_keys_match(other)
                
        for id in self._ids:
            self_params = self[id]
            other_params = other[id]

            # The parameters of the current id can either be a dictionary or None
            # so we need to check 1) if they are both dictionaries or 2) if one of them is None
            # and the other is a dictionary (which has 2 permutations) or 3) if both are None
            if isinstance(self_params, dict) and isinstance(other_params, dict):
                new_params = self._concat_two_dicts(self_params, other_params)
            elif isinstance(self_params, dict) and other_params is None:
                new_params = self._concat_dict_and_none(self_params, none_first=False)
            elif self_params is None and isinstance(other_params, dict):
                new_params = self._concat_dict_and_none(other_params, none_first=True)
            elif self_params is None and other_params is None:
                new_params = {}
                for key in keys:
                    new_params[key] = [None, None]
                
            # Update the dictionary with the new values
            self._data[id] = new_params

        self.name = f"{self.name} + {other.name}"

    def add(self, structural_parameters: dict) -> None:
        """
        Add a new set of structural parameters to the registry.
        
        :param structural_parameters: The structural parameters to add.
        :type structural_parameters: dict
        """
        # Add the corresponding id to the set
        current_id = len(self._ids)
        self._ids.add(current_id)

        # Add the data to the dictionary
        self._data[current_id] = structural_parameters

    def extend(self, other: Any) -> None:
        """
        Extend the registry with another registry.
        
        :param other: The other registry to extend with.
        :type other: StructuralParamsRegistry
        """
        # Check if the other object is of the same type
        if not isinstance(other, StructuralParamsRegistry):
            raise TypeError(
                f"Expected a StructuralParamsRegistry object, got {type(other)}"
            )
        
        if self.name != other.name:
            info = f"Warning: The registry names are different: {self.name} and {other.name}. Continuing anyway."
            print(info)

        for id in other.ids:
            params = other[id]
            self.add(params)

    def get_all(self) -> dict:
        """
        Get all the values in the registry.
        """
        return self._data
    
    def get_non_empty(self) -> dict:
        """
        Get all the non-empty values in the registry.
        """
        return {k: v for k, v in self._data.items() if v is not None}
    
    def _check_if_keys_match(self, other: Any) -> None:
        """
        Check if the keys of the two registries match.
        
        :param other: The other registry to check against.
        :type other: StructuralParamsRegistry
        """
        self_params_dict = None
        other_params_dict = None
        for id in self._ids:
            if self_params_dict is None:
                self_params = self[id]
                if self_params is not None:
                    self_params_dict = self_params

            if other_params_dict is None:
                other_params = other[id]
                if other_params is not None:
                    other_params_dict = other_params

            if isinstance(self_params_dict, dict) and isinstance(other_params_dict, dict):
                if set(self_params_dict.keys()) == set(other_params_dict.keys()):
                    keys = self_params_dict.keys()
                    del self_params_dict, other_params_dict
                    return keys
                else:
                    raise ValueError(
                        f"The two StructuralParamsRegistry objects have different keys: {self_params.keys()} and {other_params.keys()}."
                    )
                
    def _concat_two_dicts(self, self_params: dict, other_params: dict) -> dict:
        """
        Concatenate two dictionaries whose keys are the same.
        The values of the keys can be either a list or a single value.
        If both values are lists, they will be concatenated.
        If one of the values is a list and the other is a single value,
        the single value will be added to the list.
        If both values are single values, they will be concatenated into a list.
        
        :param self_params: The first dictionary.
        :type self_params: dict
        :param other_params: The second dictionary.
        :type other_params: dict
        :return: The concatenated dictionary.
        :rtype: dict
        """
        for key in self_params.keys():
            self_vals = self_params[key]
            other_vals = other_params[key]

            # Concatenate the values
            if isinstance(self_vals, list) and isinstance(other_vals, list):
                new_vals = self_vals + other_vals
                self_params[key] = new_vals
            elif isinstance(self_vals, list) and not isinstance(other_vals, list):
                new_vals = self_vals + [other_vals]
                self_params[key] = self_vals
            elif not isinstance(self_vals, list) and isinstance(other_vals, list):
                new_vals = [self_vals] + other_vals
                self_params[key] = new_vals
            else:
                new_vals = [self_vals, other_vals]
                self_params[key] = new_vals
        
        return self_params
    
    def _concat_dict_and_none(self, dict_: dict, none_first: bool) -> dict:
        """
        Concatenate a dictionary with None value.
        The values of the keys can be either a list or a single value.
        If the value is a list, None will be added to the beginning or the end of the list.
        If the value is a single value, None will be added to the beginning or the end of the list.
        If the value is None, it will be replaced with a list containing None.

        :param dict_: The dictionary to concatenate with None.
        :type dict_: dict
        :param none_first: If True, None will be added to the beginning of the list.
        :type none_first: bool
        :return: The concatenated dictionary.
        :rtype: dict
        """
        for key in dict_.keys():
            vals = dict_[key]

            if none_first:
                # Concatenate the values
                if isinstance(vals, list):
                    new_vals = [None] + vals
                    dict_[key] = new_vals
                else:
                    new_vals = [None, vals]
                    dict_[key] = new_vals
            else:
                # Concatenate the values
                if isinstance(vals, list):
                    new_vals = vals + [None]
                    dict_[key] = new_vals
                else:
                    new_vals = [vals, None]
                    dict_[key] = new_vals
        return dict_
    