import os
import pandas as pd
from scipy.io import loadmat
import numpy as np
from ..BaseDataPreparation import BaseDataPreparation

class ColonDataPreparation(BaseDataPreparation):
    def load_data(self):
        return loadmat(os.path.join(self.input_folder, "colon.mat"))

    def process_data(self):
        data_mat = self.load_data()
        
        feature_data = data_mat["X"].astype(np.float32)     # (n_samples, n_features)
        labels = pd.Series(data_mat["Y"][:, 0], name=self.label_column)  # (n_samples,)
        patients_name = pd.Series(np.arange(feature_data.shape[0]), name="cod_pz")  # (n_samples,)
        features_name = [f"feature_{i}" for i in range(feature_data.shape[1])]
        features_data = pd.DataFrame(feature_data, columns=features_name)  # (n_samples, n_features)
        data = pd.concat([patients_name, features_data, labels], axis=1)
        
        assert 'cod_pz' in data.columns, "Column 'cod_pz' not found in data"
        assert self.label_column in data.columns, f"Column {self.label_column} not found in data"

        return data
    
    
if __name__ == "__main__":
    input_folder = "/home/edofroses/genomics/Raw_Datasets"
    output_folder = "/home/edofroses/genomics/prova_colon"
    label_column = "label_colon"
    
    colon_data_prep = ColonDataPreparation(input_folder, output_folder, label_column)
    processed_data = colon_data_prep.process_data()
    print(processed_data.head())
    print(processed_data[label_column].value_counts())