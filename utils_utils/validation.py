import os
import re
import numpy as np
import torch


def validate_type(obj, expected_type, allow_none=False, obj_name="object"):
    if obj is None and allow_none:
        return
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {obj_name} to be {expected_type}, got {type(obj)}")
    
    
def validate_numpy_array(arr, ndim=None, allow_none=False):
    if arr is None and allow_none:
        return
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")
    if ndim is not None and arr.ndim not in (ndim if isinstance(ndim, (tuple, list)) else (ndim,)):
        raise ValueError(f"Expected array with ndim={ndim}, got {arr.ndim}")
    

def validate_torch_tensor(tensor, ndim=None, allow_none=False):
    if tensor is None and allow_none:
        return
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if ndim is not None and tensor.dim() not in (ndim if isinstance(ndim, (tuple, list)) else (ndim,)):
        raise ValueError(f"Expected tensor with ndim={ndim}, got {tensor.dim()}")
    

def validate_list(obj, elem_type=None, allow_none=False):
    if obj is None and allow_none:
        return
    if not isinstance(obj, list):
        raise TypeError(f"Expected list, got {type(obj)}")
    if elem_type is not None:
        for i, elem in enumerate(obj):
            if not isinstance(elem, elem_type):
                raise TypeError(f"Element {i} in list is not of type {elem_type}: {type(elem)}")
            

def validate_dict(obj, key_type=None, value_type=None, allow_none=False):
    if obj is None and allow_none:
        return
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict, got {type(obj)}")
    for k, v in obj.items():
        if key_type is not None and not isinstance(k, key_type):
            raise TypeError(f"Key {k} is not of type {key_type}: {type(k)}")
        if value_type is not None and not isinstance(v, value_type):
            raise TypeError(f"Value {v} is not of type {value_type}: {type(v)}")
        

def validate_file_exists(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    

def validate_filename_pattern(filename, pattern):
    if not re.search(pattern, filename):
        raise ValueError(f"Filename '{filename}' does not match pattern '{pattern}'")
    

########################################
############## DECORATORS ##############
########################################


def check_4tensor_inputs(func):
    def wrapper(self, x, y_pred, y_true, psf, *args, **kwargs):
        for t, name in zip([x, y_pred, y_true, psf], ['x', 'y_pred', 'y_true', 'psf']):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Input '{name}' is not a torch.Tensor")
            if t.dim() != 4:
                raise ValueError(f"Input '{name}' must be 4D (B, C, H, W), got shape {t.shape}")
        return func(self, x, y_pred, y_true, psf, *args, **kwargs)
    return wrapper
