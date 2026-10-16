import os
import time
from datetime import datetime

class LoggerCARA:
    """
    Logger per la classe CARA che registra in modo strutturato le informazioni 
    di esecuzione del processo di selezione delle caratteristiche.
    """

    def __init__(self, log_dir):
        """
        Inizializza il logger.
        
        Args:
            log_dir (str): Directory dove salvare il file di log
        """
        self.log_dir = log_dir
        self.start_time = time.time()
        self.separator = "=" * 80
        self.semi_separator = "-" * 50
        
        # Crea la directory di log se non esiste
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.log_path = os.path.join(log_dir, "featuresel_log.txt")
        
        # Inizializza il file di log con intestazione
        with open(self.log_path, "w") as f:
            f.write(f"{self.separator}\n")
            f.write(f"CARA - FEATURE SELECTION SYSTEM USING AUTOENCODER\n")
            f.write(f"{self.separator}\n\n")
            f.write(f"Esecuzione iniziata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_iteration_start(self, iteration, total_selected_features=None, selected_features_list=None):
        """
        Registra l'inizio di una nuova iterazione con dettagli.
        
        Args:
            iteration (int): Numero dell'iterazione
            total_selected_features (int, optional): Numero totale di caratteristiche selezionate finora
            selected_features_list (list, optional): Lista delle caratteristiche selezionate finora
        """
        message = (
            f"\n{self.separator}\n"
            f"ITERAZIONE {iteration} - INIZIO\n"
            f"{self.separator}\n"
        )
        
        if total_selected_features is not None:
            message += f"Caratteristiche selezionate finora: {total_selected_features}\n\n"
            
        if selected_features_list and len(selected_features_list) > 0:
            message += "Caratteristiche selezionate in precedenza:\n"
            for i, feature_group in enumerate(selected_features_list):
                message += f"  Iterazione {i}: {feature_group}\n"
            message += "\n"
        
        self.log_message(message)
    
    def log_iteration_end(self, iteration, selected_features=None):
        """
        Registra la fine di un'iterazione con i risultati.
        
        Args:
            iteration (int): Numero dell'iterazione
            selected_features (list, optional): Caratteristiche selezionate in questa iterazione
            metrics (dict, optional): Metriche di performance del modello
        """
        message = (
            f"\n{self.separator}\n"
            f"ITERAZIONE {iteration} - RISULTATI\n"
            f"{self.separator}\n"
        )
        
        if list(selected_features):
            message += f"Caratteristiche selezionate: {selected_features}\n"
            message += f"Numero di caratteristiche: {len(selected_features)}\n\n"
        
        
        message += f"{self.separator}\n"
        self.log_message(message)
    
    def log_filter_start(self, n_clusters):
        """
        Registra l'inizio della fase di autoencoder.
        
        Args:
            n_clusters (int): Numero di cluster da analizzare
        """
        message = (
            f"\nINIZIO FASE FILTRAGGIO\n"
            f"Numero di cluster da analizzare: {n_clusters}\n"
            f"{self.semi_separator}"
        )
        self.log_message(message)
    
    def log_filter_results(self, selected_features):
        """
        Registra i risultati della fase di autoencoder.
        
        Args:
            selected_features (list): Caratteristiche selezionate dall'autoencoder
        """
        if not selected_features:
            message = "\nNessuna caratteristica selezionata dal filter\n"
        else:
            message = (
                f"\nRISULTATI FILTRO\n"
                f"Caratteristiche selezionate: {len(selected_features)}\n"
                f"Lista: {selected_features}\n"
                f"{self.semi_separator}"
            )
        self.log_message(message)
    
    def log_message(self, message):
        """
        Registra un messaggio generico.
        
        Args:
            message (str): Messaggio da registrare
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        with open(self.log_path, "a") as f:
            f.write(formatted_message + '\n')
    
    def log_section(self, title):
        """
        Registra un'intestazione di sezione.
        
        Args:
            title (str): Titolo della sezione
        """
        message = (
            f"\n{'-' * 50}\n"
            f"{title}\n"
            f"{'-' * 50}\n"
        )
        self.log_message(message)
    
    def log_config(self, config):
        """
        Registra la configurazione utilizzata.
        
        Args:
            config (Config): Oggetto configurazione
        """
        message = (
            f"\n{self.separator}\n"
            f"CONFIGURAZIONE\n"
            f"{self.separator}\n"
        )
        
        # Dataset
        message += f"Dataset: {config.dataset.name}\n"
        message += f"Input folder: {config.dataset.input_folder}\n"
        message += f"Label column: {config.dataset.label_column}\n"
        message += f"Scaler: {config.dataset.scaler}\n\n"
        
        # Clustering
        message += f"Clustering strategy: {config.clustering.strategy}\n"
        message += f"Clustering parameters: {config.clustering.parameters}\n\n"
        
        # Autoencoder
        message += f"Autoencoder: {config.filters.autoencoder_parameters.AE_name}\n"
        message += f"Learning rate: {config.filters.autoencoder_parameters.lr}\n"
        message += f"Epochs: {config.filters.autoencoder_parameters.epochs}\n\n"
        
        # Output
        message += f"Output folder: {config.output.output_folder}\n"
        
        message += f"{self.separator}\n"
        self.log_message(message)
    
    def log_completion(self):
        """
        Registra il completamento dell'esecuzione con il tempo impiegato.
        """
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        
        message = (
            f"\n{self.separator}\n"
            f"ESECUZIONE COMPLETATA\n"
            f"{self.separator}\n"
            f"Data e ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Tempo di esecuzione: {hours}h {minutes}m {seconds:.2f}s\n"
            f"{self.separator}\n"
        )
        
        self.log_message(message)