from data_loader.csv_data import CSVData

DATA_PATH = r'C:\Users\simpe\OneDrive\Documents\DATA\global_disaster_response_2018_2024.csv'

data = CSVData(path=DATA_PATH, target_col="recovery_days")
print(data)