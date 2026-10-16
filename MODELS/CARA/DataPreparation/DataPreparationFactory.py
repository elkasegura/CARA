import os
import pandas as pd
from .dataset_registry import DATASET_CLASSES

class DataPreparationFactory:
    DATASET_CLASSES = DATASET_CLASSES
    
    def __init__(self, dataset_name, input_folder, output_folder, label_column=None):
        """
        Initialize the DataPreparationFactory.

        :param dataset_name: Name of the dataset.
        :param input_folder: Path to the input folder containing the dataset.
        :param output_folder: Path to the output folder where results will be saved.
        :param label_column: Name of the column containing labels.
        """
        self.dataset_name = dataset_name
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.label_column = label_column
        self.label_balance_string = 'Label balance: '
        self.export = True
        
        # File richiesti nella cartella di output
        self.required_files = ['data.csv', 'features_data.csv', 'features.csv', 'labels.csv']
        
    def _check_output_folder(self):
        """
        Verifica se la cartella di output contiene già tutti i file necessari del dataset.
        Se tutti i file esistono, imposta export = False, altrimenti export = True.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Verifica se tutti i file richiesti esistono
        all_files_exist = all(
            os.path.exists(os.path.join(self.output_folder, filename)) 
            for filename in self.required_files
        )
        
        if all_files_exist:
            print(f"Dati già presenti nella cartella {self.output_folder}. Caricamento in corso...")
            self.export = False
        else:
            print(f"Dati non trovati nella cartella {self.output_folder}. Preparazione e salvataggio in corso...")
            self.export = True
                    
    def _save_dataset_info(self): 
        """Salva le informazioni del dataset in un file di testo."""
        info_file_path = os.path.join(self.output_folder, 'dataset_info.txt')
        with open(info_file_path, 'w') as file:
            file.write(f"Dataset Name: {self.dataset_name}\n")
            file.write(f"Label Column: {self.label_column}\n")
            file.write(self.label_balance_string + "\n")
            
    def _save_data(self, data, features_data, features, labels):
        """Salva i dati processati nei file CSV."""
        data.to_csv(os.path.join(self.output_folder, "data.csv"), index=False)
        features_data.to_csv(os.path.join(self.output_folder, "features_data.csv"), index=False)
        pd.Series(features).to_csv(os.path.join(self.output_folder, "features.csv"), index=False, header=False)
        labels.to_csv(os.path.join(self.output_folder, "labels.csv"), index=False, header=False)
        print(f"Dati salvati con successo in {self.output_folder}")
            
    def _prepare_data_training(self, data):
        """
        Prepara i dati per il training separando features e labels.
        
        :param data: DataFrame contenente i dati completi.
        :return: Tuple con (data, features_data, features, labels).
        """
        labels = data[self.label_column].copy()
    
        #if not set(labels.unique()).issubset({0, 1}):
        #    raise ValueError("I label devono essere 0 o 1.")
        
        #self.label_balance_string = f"Label counts - class 0: {(labels == 0).sum()}, class 1: {(labels == 1).sum()}"
        self.label_balance_string = "Label counts - " + ", ".join([f"class {i} : {(labels == i).sum()}" for i in labels.unique()])
        
        features_data = data.drop(['cod_pz', self.label_column], axis=1).copy()
        features = features_data.columns.values
        
        return data, features_data, features, labels

    def export_data(self):
        """
        Esporta o carica i dati del dataset.
        Se i dati esistono già, li carica. Altrimenti li processa e li salva.
        
        :return: Dictionary contenente data, features_data, features e labels.
        """
        self._check_output_folder()
        
        if not self.export:
            # Carica i dati esistenti
            data = pd.read_csv(os.path.join(self.output_folder, "data.csv"))
            data, features_data, features, labels = self._prepare_data_training(data)
        else:
            # Processa e salva nuovi dati
            preparation_class = self.DATASET_CLASSES.get(self.dataset_name)
            
            if preparation_class is None:
                raise ValueError(f"Dataset sconosciuto: {self.dataset_name}")
            
            data_preparation = preparation_class(
                self.input_folder, 
                self.output_folder, 
                label_column=self.label_column
            )
            
            data = data_preparation.process_data()
            data, features_data, features, labels = self._prepare_data_training(data)
            
            self._save_dataset_info()
            self._save_data(data, features_data, features, labels)
                
        return {
            "data": data,
            "features_data": features_data,
            "features": features,
            "labels": labels
        }
            
if __name__ == "__main__":
    
    # Carlo_dataset
    input_folder = "/home/edofroses/genomics/Raw_Datasets/Carlo_data"
    output_folder = "/home/edofroses/genomics/dataset_training_kmeans"
    label_column = "TTFT"
    
    data_preparation_factory = DataPreparationFactory("Leukemia", input_folder, output_folder, label_column)
    data_preparation_factory.export_data()
    
    """
    # leukemia_ALL_AML
    input_folder = "data/leukemia_ALL_AML"
    output_folder = "data/leukemia_ALL_AML_output"
    cluster_strategy = "hierarchical"
    label_column = "cancer"
    
    data_preparation_factory = DataPreparationFactory("leukemia_ALL_AML", input_folder, output_folder, cluster_strategy, label_column)
    data_preparation_factory.export_data_clusters()
    
    # leukemia_ALL_AML
    input_folder = "data/leukemia_ALL_AML"
    output_folder = "data/leukemia_ALL_AML_output"
    cluster_strategy = "correlation"
    label_column = "cancer"
    
    data_preparation_factory = DataPreparationFactory("leukemia_ALL_AML", input_folder, output_folder, cluster_strategy, label_column)
    data_preparation_factory.export_data_clusters()
    """