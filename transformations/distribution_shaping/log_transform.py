from transformations.feature_transformer import BaseFeatureTransformer
import numpy as np
import pandas as pd

class LogTransform(BaseFeatureTransformer):
    """
    Apply a log(1 + x) transformation to reduce skewness of numeric features.
    Handles non-negative values safely.
    """

    def __init__(self, features=None, offset=1e-6):
        """
        features: list of feature names to apply to (None = all numeric features)
        offset: small constant added to avoid log(0)
        """
        super().__init__(features=features, name="log_transform", offset=offset)

    def transform_feature(self, series: pd.Series) -> pd.Series:
        offset = self.params["offset"]

        # Ensure we don't apply log to negative values
        if (series < -offset).any():
            raise ValueError(
                f"Feature '{series.name}' contains negative values, cannot apply log transform safely."
            )

        return np.log1p(series + offset)
