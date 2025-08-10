from .tester import UniversalTester
from . import strategies
from . import metrics

# Import modules to execute decorators
from .strategies.strategies import *
from .metrics import *

__all__ = ['UniversalTester']
