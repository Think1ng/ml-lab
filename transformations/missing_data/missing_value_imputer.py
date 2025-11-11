from transformations.feature_transformer import BaseFeatureTransformer
import pandas as pd
import numpy as np


class MissingValueImputer(BaseFeatureTransformer):
    """
    Impute missing values in features using a specified strategy.

    Supported strategies:
      - "mean": replace with column mean (numeric only)
      - "median": replace with column median (numeric only)
      - "mode": replace with most frequent value
      - "constant": replace with a given constant value
      - "ffill": forward fill (good for time series)
      - "bfill": backward fill
    """

    def __init__(self, features=None, strategy="mean", fill_value=None):
        """
        Parameters
        ----------
        features : list[str] or None
            Which features to apply the imputer to (None = all features)
        strategy : str
            One of ["mean", "median", "mode", "constant", "ffill", "bfill"]
        fill_value : any
            Value to use if strategy="constant"
        """
        super().__init__(
            features=features,
            name=f"missing_imputer_{strategy}",
            strategy=strategy,
            fill_value=fill_value
        )

    def transform_feature(self, series: pd.Series) -> pd.Series:
        strategy = self.params["strategy"]
        fill_value = self.params["fill_value"]

        n_missing = series.isna().sum()
        if n_missing == 0:
            return series  # nothing to impute

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise TypeError(f"Cannot use mean imputation on non-numeric feature '{series.name}'")
            value = series.mean()

        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                raise TypeError(f"Cannot use median imputation on non-numeric feature '{series.name}'")
            value = series.median()

        elif strategy == "mode":
            value = series.mode(dropna=True)
            value = value.iloc[0] if not value.empty else np.nan

        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("Must provide 'fill_value' when using strategy='constant'")
            value = fill_value

        elif strategy == "ffill":
            return series.ffill()

        elif strategy == "bfill":
            return series.bfill()

        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        return series.fillna(value)
