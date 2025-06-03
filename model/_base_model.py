from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all models.
    This class defines the basic structure and methods
    that all models should implement.
    """
    @abstractmethod
    def train_model(self):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Save the model.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Load the model.
        """
        pass
