from transformations.feature_transformer import BaseFeatureTransformer

class StandardZScore(BaseFeatureTransformer):
    def __init__(self, features=None):
        super().__init__(features=features, name="standard_zscore")

    def transform_feature(self, series):
        mean = series.mean()
        std = series.std()
        return (series - mean) / std
