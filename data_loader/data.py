from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Optional, List, Union

class Data(ABC):
    """
    Abstract base class for loading and managing datasets.

    Attributes
    ----------
    df : pd.DataFrame
        The full dataset containing both features and target.
    target_col : str
        Name of the target column.
    X : pd.DataFrame
        Feature matrix (all columns except target_col).
    y : pd.Series
        Target vector (the column specified by target_col).
    name : Optional[str]
        Optional name identifier for the dataset.
    """

    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col

        self._validate_target_exists()
        self.X, self.y = self._split_X_y()

    def _validate_target_exists(self):
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame columns.")

    def _split_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return X, y

    @abstractmethod
    def load(self):
        """Load the dataset (to be implemented by subclasses)."""
        pass

    @abstractmethod
    def info(self) -> str:
        """Return a short summary about the dataset."""
        pass


    def __repr__(self):
        lines = []
        n_rows, n_cols = self.df.shape
        lines.append(f"Data: {n_rows} datapoints, {n_cols} features\n")

        for col in self.X.columns:
            series = self.X[col]
            dtype = series.dtype
            n_missing = series.isna().mean() * 100
            max_val = series.max() if pd.api.types.is_numeric_dtype(series) else None
            min_val = series.min() if pd.api.types.is_numeric_dtype(series) else None
            mean_val = series.mean() if pd.api.types.is_numeric_dtype(series) else None
            median_val = series.median() if pd.api.types.is_numeric_dtype(series) else None

            line = f"- {col} ({dtype}): NaN={n_missing:.1f}%"
            if pd.api.types.is_numeric_dtype(series):
                line += f", max={max_val:.2f}, min={min_val:.2f}, mean={mean_val:.2f}, median={median_val:.2f}"
            lines.append(line)

        return "\n".join(lines)

