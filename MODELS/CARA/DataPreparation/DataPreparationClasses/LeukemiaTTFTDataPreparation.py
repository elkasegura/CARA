import os
import pandas as pd
from ..BaseDataPreparation import BaseDataPreparation

class LeukemiaTTFTDataPreparation(BaseDataPreparation):
    def load_data(self):
        return pd.read_csv(os.path.join(self.input_folder, "data_merged.csv"))

    def process_data(self):
        data = self.load_data()
        data = data[data['evento'] == 1].copy()
        data.drop(columns=['evento'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[self.label_column] = data[self.label_column].apply(lambda x: 0 if x < 24 else 1)
        
        assert 'cod_pz' in data.columns, "Column 'cod_pz' not found in data"
        assert self.label_column in data.columns, f"Column {self.label_column} not found in data"

        return data