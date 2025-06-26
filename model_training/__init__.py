"""
NOTE FOR USERS:

STILL IN DEVELOPMENT.
This module provides tools for training models in the DRAGN project.
"""

from .trainer import Trainer
from .loss_functions import (
    _get_avaliable_loss_funcstions,
    get_loss_function
)


AVALIABLE_LOSS_FUNCTIONS = list(_get_avaliable_loss_funcstions().keys())
