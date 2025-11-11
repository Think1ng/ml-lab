from data_loader.csv_data import CSVData
from transformations.zscore.standard_zscore import StandardZScore

DATA_PATH = r'C:\Users\simpe\OneDrive\Documents\DATA\global_disaster_response_2018_2024.csv'

# First load the data using CSVData
data = CSVData(path=DATA_PATH, target_col="recovery_days")
print(data)

# Then create a StandardZScoreTransformer and transform the data
transformer = StandardZScore(features=["response_efficiency_score", "economic_loss_usd"])
transformer.transform(data) # it transforms data in place

print(data)