from abc import ABC, abstractmethod
from data_loader.data import Data
from typing import List

class BaseFeatureTransformer(ABC):
    """
    Abstract base class for any feature-level transformation.
    Handles feature selection, naming, and parameter passing.
    """
    def __init__(self, features: List[str] = None, name: str = None, **params):
        """
        features: list of columns to apply transformation on. None = all numeric features.
        name: optional suffix for transformed Data name
        params: custom parameters for the transformation (e.g., rolling window)
        """
        self.features = features
        self.name = name
        self.params = params

    @abstractmethod
    def transform_feature(self, series):
        """Implement the actual feature transformation on a pandas Series"""
        pass

    def transform(self, data: Data):
        """
        Apply transformation to all selected features and return a new Data object
        """
        features_to_use = self.features
        for f in features_to_use:
            data.df[f] = self.transform_feature(data.df[f])
