# data_loader/csv_data.py
import pandas as pd
from .data import Data

class CSVData(Data):
    """Concrete class for loading data from CSV files."""

    def __init__(self, path: str, target_col: str):
        self.path = path
        df = self.load()
        super().__init__(df=df, target_col=target_col)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
