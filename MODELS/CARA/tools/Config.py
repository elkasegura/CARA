import yaml
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    name: str
    input_folder: str
    label_column: str
    scaler: str

@dataclass
class OutputConfig:
    output_folder: str
    logs_folder: str

class ClusteringConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.strategy = config_dict.get('strategy', 'KMeans')
        self.cluster_on_correlation = config_dict.get('cluster_on_correlation', True)
        self.number_of_clusters = config_dict.get('number_of_clusters', None)
        self.range_n_clusters = config_dict.get('range_n_clusters', 0)
        self._parameters = config_dict.get('parameters', "searching")
    
    @property
    def parameters(self) -> Union[str, Dict[str, Any]]:
        """
        Restituisce i parametri di clustering che possono essere:
        - La stringa "searching"
        - Un dizionario di parametri
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, value: Union[str, Dict[str, Any]]):
        """
        Imposta i parametri di clustering.
        
        Args:
            value: Può essere la stringa "searching" o un dizionario di parametri
        """
        if not isinstance(value, (str, dict)):
            raise TypeError("I parametri di clustering devono essere una stringa o un dizionario")
        
        if isinstance(value, str) and value != "searching":
            raise ValueError('Se i parametri sono una stringa, deve essere "searching"')
            
        self._parameters = value

@dataclass
class AutoencoderConfig:
    lr: float = 0.001
    epochs: int = 100
    AE_name: str = 'LinearAE'
    n_layers: int = 3

@dataclass
class MedoidConfig:
    distance_metric: str = 'euclidean'

@dataclass
class FiltersConfig:
    filter_type: str = 'AE'
    autoencoder_parameters: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    medoid_parameters: MedoidConfig = field(default_factory=MedoidConfig)


class Config:
    def __init__(self, config_path: str):
        """
        Inizializza l'oggetto Config leggendo il file YAML.
        
        Args:
            config_path: Percorso del file di configurazione YAML
        """
        self._config_dict = self._load_config(config_path)
        
        # Inizializzazione delle sezioni principali
        self.dataset = DatasetConfig(**self._config_dict.get('dataset', {}))
        self.output = OutputConfig(**self._config_dict.get('output', {}))
        
        # Per clustering utilizziamo una classe personalizzata che gestisce sia "searching" che dizionari
        clustering_dict = self._config_dict.get('clustering', {})
        self.clustering = ClusteringConfig(clustering_dict)
        
        
        filters_dict = self._config_dict.get('filters', {})
        # Converti i dizionari nested in oggetti Config
        if 'autoencoder_parameters' in filters_dict:
            filters_dict['autoencoder_parameters'] = AutoencoderConfig(**filters_dict['autoencoder_parameters'])
        if 'medoid_parameters' in filters_dict:
            filters_dict['medoid_parameters'] = MedoidConfig(**filters_dict['medoid_parameters'])

        self.filters = FiltersConfig(**filters_dict)
        
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carica il file di configurazione YAML.
        
        Args:
            config_path: Percorso del file YAML
            
        Returns:
            Dizionario di configurazione
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Errore nel caricamento del file di configurazione: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte la configurazione in un dizionario.
        
        Returns:
            Dizionario della configurazione completa
        """
        return self._config_dict

    def save(self, path: str) -> None:
        """
        Salva la configurazione in un file YAML.
        
        Args:
            path: Percorso dove salvare il file
        """
        with open(path, 'w') as file:
            yaml.dump(self._config_dict, file, default_flow_style=False)
            
    def validate(self) -> bool:
        """
        Valida che la configurazione contenga tutti i parametri necessari
        e che siano di tipo corretto.
        
        Returns:
            True se la configurazione è valida
            
        Raises:
            ValueError: Se la configurazione non è valida
        """
        # Verifica i valori obbligatori
        if not hasattr(self.dataset, 'input_folder') or not self.dataset.input_folder:
            raise ValueError("Manca il parametro obbligatorio 'input_folder' in dataset")
            
        if not hasattr(self.output, 'output_folder') or not self.output.output_folder:
            raise ValueError("Manca il parametro obbligatorio 'output_folder' in output")
            
        # Verifica la strategia di clustering
        valid_strategies = ['KMeans', 'hierarchical', 'DBScan', 'KMedoids']
        if self.clustering.strategy not in valid_strategies:
            raise ValueError(f"La strategia di clustering '{self.clustering.strategy}' non è valida. "
                            f"Deve essere una tra: {', '.join(valid_strategies)}")
        
        return True