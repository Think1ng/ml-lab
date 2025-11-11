from transformations.feature_transformer import BaseFeatureTransformer
import pandas as pd

class Winsorize(BaseFeatureTransformer):
    """
    Winsorize numeric features by capping values at lower/upper quantiles.
    """
    def __init__(self, features=None, lower_quantile=0.01, upper_quantile=0.99):
        """
        features: list of feature names to apply to (None = all numeric features)
        lower_quantile: lower percentile to cap (e.g., 0.01)
        upper_quantile: upper percentile to cap (e.g., 0.99)
        """
        super().__init__(features=features, name=f"winsorize_{lower_quantile * 100}_{upper_quantile * 100}",
                         lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    def transform_feature(self, series: pd.Series) -> pd.Series:
        lower = series.quantile(self.params["lower_quantile"])
        upper = series.quantile(self.params["upper_quantile"])
        return series.clip(lower=lower, upper=upper)