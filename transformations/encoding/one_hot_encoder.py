from transformations.feature_transformer import BaseFeatureTransformer
import pandas as pd


class OneHotEncoder(BaseFeatureTransformer):
    """
    Apply one-hot encoding to categorical features.
    Expands each categorical column into multiple binary indicator columns.
    """

    def __init__(self, features=None, drop_first=False, prefix_sep="_"):
        """
        Parameters
        ----------
        features : list[str] or None
            List of categorical feature names to encode (None = auto-detect categorical columns)
        drop_first : bool
            Whether to drop the first category (avoids multicollinearity in linear models)
        prefix_sep : str
            Separator between feature name and category in new column names
        """
        super().__init__(
            features=features,
            name="one_hot_encoder",
            drop_first=drop_first,
            prefix_sep=prefix_sep
        )
        self.generated_columns = []  # store new columns for reference

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Determine which features to encode
        features = self.features

        drop_first = self.params["drop_first"]
        prefix_sep = self.params["prefix_sep"]

        # One-hot encode selected features
        X_encoded = pd.get_dummies(
            X,
            columns=features,
            drop_first=drop_first,
            prefix_sep=prefix_sep,
            dtype=float
        )


        return X_encoded
