import os
import pandas as pd

class CSVDataHandler:
    def __init__(self, directory_path, type):
        self.directory_path = directory_path
        self.type = type
        self.dataframes = {}

    def load_csv_files(self):
        files_in_directory = os.listdir(self.directory_path)
        csv_files = [file for file in files_in_directory if file.endswith('.csv')]

        for csv_file in csv_files:
            full_file_path = os.path.join(self.directory_path, csv_file)
            df = pd.read_csv(full_file_path)
            self.dataframes[csv_file] = df

    def get_dataframe(self, participantID):
        if self.type == 'train':
            return self.dataframes.get(str(participantID)+'-ws-training_processed.csv', None)
        if self.type == 'test':
            return self.dataframes.get(str(participantID)+'-ws-testing_processed.csv', None)

    def handle_missing_values(self, channels, fill_types):
        for channel, fill_type in zip(channels, fill_types):
            for df in self.dataframes.values():
                if isinstance(fill_type, str):
                    df[channel].interpolate(method=fill_type, inplace=True)
                    df[channel].fillna(method='ffill', inplace=True)
                    df[channel].fillna(method='bfill', inplace=True)
                else:
                    df[channel].fillna(fill_type, inplace=True)
        for df in self.dataframes.values():
            df.dropna(subset=channels, inplace=True)

    def get_combined_df(self):
        return pd.concat(self.dataframes.values())

