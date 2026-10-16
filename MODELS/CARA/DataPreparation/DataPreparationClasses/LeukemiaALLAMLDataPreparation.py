import os
import pandas as pd
from ..BaseDataPreparation import BaseDataPreparation

class LeukemiaALLAMLDataPreparation(BaseDataPreparation):
    def __init__(self, input_folder, output_folder, cluster_strategy, cluster_parameters, label_column='cancer'):
        super().__init__(input_folder, output_folder, cluster_strategy, cluster_parameters, label_column)
        self.labels_name = 'actual.csv'
        self.data_name_train = 'data_set_ALL_AML_train.csv'
        self.data_name_test = 'data_set_ALL_AML_independent.csv'

    def load_data(self):
        data_train = pd.read_csv(os.path.join(self.input_folder, self.data_name_train))
        data_test = pd.read_csv(os.path.join(self.input_folder, self.data_name_test))
        data_labels = pd.read_csv(os.path.join(self.input_folder, self.labels_name), index_col='patient')
        return data_labels, data_train, data_test

    def _process_call_columns(self, df):
        call_cols = [col for col in df.columns if 'call' in col]
        df_call = df[call_cols].replace({"A": 0, "P": 1, "M": 2})
        df.drop(call_cols, axis=1, inplace=True)
        return df, df_call

    def _extract_patient_ids(self, df):
        return list(map(int, list(df.columns[2:])))

    def _transpose_call_matrix(self, df_call, ids):
        df_call.columns = ids
        return df_call.T

    def _map_labels(self, ids, labels_df):
        labels = [labels_df.loc[i, 'cancer'] for i in ids]
        return pd.DataFrame({"patient": ids, "cancer": labels})

    def _process_numerical_features(self, X_train, X_test, train_ids, test_ids):
        gene_names = X_train.iloc[:, 1]
        X_train_processed = X_train.iloc[:, 2:].T
        X_train_processed.columns = gene_names
        X_train_processed.index = train_ids

        X_test_processed = X_test.iloc[:, 2:].T
        X_test_processed.columns = gene_names
        X_test_processed.index = test_ids

        return X_train_processed, X_test_processed

    def get_data(self):
        data_labels, data_train, data_test = self.load_data()
        data_train, data_train_call = self._process_call_columns(data_train)
        data_test, data_test_call = self._process_call_columns(data_test)

        train_ids = self._extract_patient_ids(data_train)
        test_ids = self._extract_patient_ids(data_test)

        data_train_call = self._transpose_call_matrix(data_train_call, train_ids)
        data_test_call = self._transpose_call_matrix(data_test_call, test_ids)

        train_labels = self._map_labels(train_ids, data_labels)
        test_labels = self._map_labels(test_ids, data_labels)

        train_labels['cancer'] = train_labels['cancer'].replace({"ALL": 0, "AML": 1})
        test_labels['cancer'] = test_labels['cancer'].replace({"ALL": 0, "AML": 1})

        data_train_processed, data_test_processed = self._process_numerical_features(data_train, data_test, train_ids, test_ids)
        train_combined = pd.concat([data_train_processed, train_labels.set_index('patient')], axis=1)
        test_combined = pd.concat([data_test_processed, test_labels.set_index('patient')], axis=1)

        final_data = pd.concat([train_combined, test_combined])
        final_data = final_data.reset_index().rename(columns={'index': 'cod_pz'})
        return final_data

    def process_data(self):
        data = self.get_data()

        assert 'cod_pz' in data.columns, "Column 'cod_pz' not found in data"
        assert self.label_column in data.columns, f"Column {self.label_column} not found in data"

        return data