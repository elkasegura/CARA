import os
import time
from datetime import datetime

class FilterLogger:
    """
    Logger per il filtro che registra le informazioni di selezione delle features.
    """
    
    def __init__(self, iteration, output_dir='best_models'):
        """
        Inizializza il logger.
        
        Args:
            iteration (int/str): Iterazione corrente o identificatore (es. 'WINNER')
            output_dir (str): Directory base dove salvare i log
        """
        self.model_dir = os.path.join(output_dir, f'output_iteration_{iteration}')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.log_path = os.path.join(self.model_dir, 'filter_log.txt')
        self.start_time = time.time()
        self.separator = "=" * 50
        
        # Intestazione del file di log
        with open(self.log_path, 'w') as f:
            f.write(f"=== FILTER LOG - ITERAZIONE {iteration} ===\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_message(self, message):
        """Registra un messaggio generico nel log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        with open(self.log_path, 'a') as f:
            f.write(f"{formatted_message}\n")
        print(formatted_message)
    
    def log_cluster_start(self, cluster_id, num_features):
        """
        Registra l'inizio dell'addestramento per un cluster.
        
        Args:
            cluster_id (int): ID del cluster
            num_features (int): Numero di feature nel cluster
        """
        message = (
            f"\n{self.separator}\n"
            f"CLUSTER {cluster_id}\n"
            f"{self.separator}\n"
            f"Numero di feature (campioni diventano features): {num_features}\n"
        )
        self.log_message(message)
    
    def log_training_progress(self, epoch, total_epochs, loss, accuracy, lr=None):
        """
        Registra il progresso dell'addestramento.
        
        Args:
            epoch (int): Epoca corrente
            total_epochs (int): Numero totale di epoche
            loss (float): Valore della loss
            accuracy (float): Accuratezza (errore MAE)
            lr (float, optional): Learning rate corrente
        """
        # Registra ogni 10 epoche o all'ultima epoca
        if epoch % 10 == 0 or epoch == total_epochs - 1:
            message = f"Epoca: {epoch+1}/{total_epochs}, Loss: {loss:.6f}, Errore MAE: {accuracy:.6f}"
            if lr is not None:
                message += f", LR: {lr:.6e}"
            self.log_message(message)
    
    def log_feature_selection(self, cluster_id, selected_feature, error):
        """
        Registra la feature selezionata per un cluster.
        
        Args:
            cluster_id (int): ID del cluster
            selected_feature (str): Nome della feature selezionata
            error (float): Errore di ricostruzione della feature
        """
        message = (
            f"Feature selezionata per cluster {cluster_id}: {selected_feature}\n"
            f"Errore di ricostruzione: {error:.6f}\n"
            f"{self.separator}\n"
        )
        self.log_message(message)
    
    def log_completion(self):
        """Registra il completamento dell'esecuzione con il tempo impiegato."""
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        
        message = (
            f"\n{self.separator}\n"
            "RIEPILOGO ESECUZIONE FILTER\n" + 
            f"{self.separator}\n"
            f"Tempo di esecuzione: {minutes} minuti e {seconds:.2f} secondi\n"
            f"{self.separator}\n"
        )
        
        self.log_message(message)
    
    def log_selected_features_summary(self, selected_features):
        """
        Registra un riepilogo delle feature selezionate.
        
        Args:
            selected_features (list): Lista delle feature selezionate
        """
        message = (
            f"\n{self.separator}\n"
            "RIEPILOGO FEATURES SELEZIONATE\n" + 
            f"{self.separator}\n"
            f"Totale features selezionate: {len(selected_features)}\n\n"
        )
        
        for i, feature in enumerate(selected_features, 1):
            message += f"{i}. {feature}\n"
        
        message += f"{self.separator}\n"
        self.log_message(message)