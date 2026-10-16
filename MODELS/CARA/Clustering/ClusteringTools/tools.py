import logging
import os
from datetime import datetime
from sklearn.metrics import silhouette_samples, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform, mahalanobis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import rankdata

class ClusteringLogger:
    def __init__(self, log_filename=None, log_dir="logs"):
        """
        Inizializza il logger per un determinato run.
        Se `log_filename` è specificato, lo usa come nome del file.
        Altrimenti, crea un nome con timestamp per evitare sovrascritture.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Crea la cartella logs se non esiste

        if log_filename is None:
            log_filename = f"clustering_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        self.log_filepath = os.path.join(log_dir, log_filename)

        # Configura il logger
        self.logger = logging.getLogger(self.log_filepath)
        self.logger.setLevel(logging.INFO)

        # Evita di aggiungere più handler se il logger esiste già
        if not self.logger.handlers:
            # Handler per il file di log
            file_handler = logging.FileHandler(self.log_filepath, mode="w")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)

            # Handler per la console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(console_formatter)

            # Aggiungi gli handler al logger
            self.logger.addHandler(file_handler)
            #self.logger.addHandler(console_handler)

        self.logger.info(f"Inizio log: {self.log_filepath}")

    def log(self, message, level="info"):
        """Logga un messaggio con il livello specificato."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        if level == "info":
            self.logger.info(formatted_message)
        elif level == "warning":
            self.logger.warning(formatted_message)
        elif level == "error":
            self.logger.error(formatted_message)
        elif level == "critical":
            self.logger.critical(formatted_message)
        elif level == "debug":
            self.logger.debug(formatted_message)

    def get_log_filepath(self):
        """Ritorna il percorso del file di log corrente."""
        return self.log_filepath

def evaluate_cluster(df, labels, method, min_element_per_cluster):
    """Valuta la qualità del clustering con il metodo scelto"""
    
    labels = np.array(labels)
    df_T = df.T  # per lavorare su feature (colonne)
    
    # Conta occorrenze per ogni cluster
    unique_clusters, counts = np.unique(labels, return_counts=True)
    cluster_size_dict = dict(zip(unique_clusters, counts))

    # Seleziona i cluster validi (quelli con dimensione > min_element_per_cluster)
    valid_clusters = [c for c in unique_clusters if cluster_size_dict[c] > min_element_per_cluster]
    mask = np.isin(labels, valid_clusters)
    
    avg_features_per_cluster = np.mean(counts[counts > min_element_per_cluster]) if np.any(counts > min_element_per_cluster) else 1
    single_feature_clusters = np.sum(counts <= min_element_per_cluster)

    if len(valid_clusters) < 2:
        return -1, avg_features_per_cluster, single_feature_clusters  # Penalizzazione: clustering non valido

    if method == 'silhouette':
        scores = silhouette_samples(df_T[mask], labels[mask])
        return np.mean(scores), avg_features_per_cluster, single_feature_clusters

    elif method == 'davies-bouldin':
        return davies_bouldin_score(df_T[mask], labels[mask]), avg_features_per_cluster, single_feature_clusters

    return -1, avg_features_per_cluster, single_feature_clusters

import numpy as np

def remap_labels_distinct_noise(labels):
    """
    Riassegna i label:
    - I cluster esistenti (diversi da -1) ricevono ID consecutivi a partire da 0.
    - Ogni punto con label -1 riceve un ID univoco successivo ai cluster normali.
    """
    labels = np.array(labels)
    new_labels = labels.copy()

    # Etichette dei cluster "veri" (diversi da -1)
    valid_mask = labels != -1
    unique_valid = np.unique(labels[valid_mask])
    label_map = {old: new for new, old in enumerate(unique_valid)}
    
    # Riassegna i cluster "veri"
    new_labels[valid_mask] = [label_map[l] for l in labels[valid_mask]]

    # Assegna ID unici ai punti rumore (-1)
    noise_indices = np.where(labels == -1)[0]
    next_id = len(unique_valid)
    for i, idx in enumerate(noise_indices):
        new_labels[idx] = next_id + i

    return new_labels

def select_best_clustering(dict_results_list):
    """
    Seleziona la migliore configurazione di clustering seguendo questi criteri:
    1. Ordina la lista per 'single_feature_clusters_percentage' crescente.
    2. Seleziona la migliore configurazione che soddisfa:
       - Silhouette Score > 0.25
       - Davies-Bouldin Score < 1.5
    3. Se nessuna configurazione soddisfa i vincoli, sceglie quella con:
       - Silhouette Score massimo
       - Davies-Bouldin Score minimo
    """
    # Ordina la lista per percentuale minima di single feature clusters
    sorted_results = sorted(dict_results_list, key=lambda x: x['single_feature_clusters_percentage'])

    # Filtra le configurazioni che rispettano i vincoli
    valid_results = [
        result for result in sorted_results
        if result['silhouette_score'] > 0.25 and result['davies_bouldin_score'] < 1.5
    ]

    # Se ci sono risultati validi, scegli il migliore tra quelli
    if valid_results:
        return valid_results[0]  # Il primo della lista ordinata (quello con meno single-feature clusters)

    # Se non ci sono risultati validi, scegliere la migliore configurazione possibile
    best_fallback = min(
        sorted_results,
        key=lambda x: (-x['silhouette_score'], x['davies_bouldin_score'])  # Silhouette più alta e Davies più basso
    )

    return best_fallback

def compute_feature_distances(df, metric='euclidean'):
    # Seleziona le sole colonne numeriche
    numeric_df = df.select_dtypes(include='number')

    if np.any(np.isnan(numeric_df.values)) or np.any(np.isinf(numeric_df.values)):
        raise ValueError("⚠️ I dati contengono NaN o Inf. Devi correggerli prima di calcolare la distanza.")
    
    # Normalizziamo se necessario
    if metric == 'euclidean':
        scaler = StandardScaler()
        numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
        dist_array = pdist(numeric_df.T, metric=metric)
    elif metric == 'pearson':
        corr_matrix = np.corrcoef(numeric_df.T)
        dist_array = squareform(1 - np.abs(corr_matrix), checks=False)
    elif metric == 'spearman':
        numeric_df_ranked = numeric_df.apply(rankdata, axis=0)
        # Calcola correlazione tra colonne (feature)
        corr_matrix = np.corrcoef(numeric_df_ranked, rowvar=False)
        # Rendi la matrice esplicitamente simmetrica
        corr_sym = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_sym, 1.0)
        dist_array = squareform(1 - np.abs(corr_sym), checks=False)
    elif metric == 'cosine':
        # Normalizzazione L2 delle colonne per cosine similarity
        normalized_df = numeric_df / np.linalg.norm(numeric_df, axis=0, keepdims=True)
        cosine_similarity = np.dot(normalized_df.T, normalized_df)
        dist_array = squareform(1 - np.abs(cosine_similarity), checks=False)
    else:
        raise ValueError(f"⚠️ Metrica non supportata: {metric}")
    
    num_nans = np.isnan(dist_array).sum()

    if num_nans > 0:
        print(f"⚠️ ATTENZIONE: {num_nans} NaN trovati nella matrice delle distanze! → Feature messe nello stesso cluster.")
        dist_array = np.nan_to_num(dist_array, nan=0.0)  # Sostituisci i NaN con 0
    
    num_infs = np.isinf(dist_array).sum()
    if num_infs > 0:
        print(f"⚠️ ATTENZIONE: Trovati {num_infs} valori Inf nella matrice delle distanze!")
        dist_array = np.nan_to_num(dist_array, posinf=1e9, neginf=-1e9)  # Sostituisci gli Inf con valori molto grandi
    
    if np.any(np.isinf(dist_array)):
        raise ValueError("⚠️ Errore: La matrice delle distanze contiene valori Inf.")

    return dist_array  # Restituisci la matrice condensata

def cosine_distance(point1, point2):
    """!
    @brief Calculate cosine distance between two vectors.

    @param[in] point1 (array_like): First vector.
    @param[in] point2 (array_like): Second vector.

    @return (float) Cosine distance between the two vectors.
    """
    dot_product = np.dot(point1, point2)
    norm_a = np.linalg.norm(point1)
    norm_b = np.linalg.norm(point2)
    return 1.0 - np.abs((dot_product / (norm_a * norm_b)))

def pearson_correlation_distance(point1, point2):
    """!
    @brief Calculate Pearson correlation distance between two vectors.

    @param[in] point1 (array_like): First vector.
    @param[in] point2 (array_like): Second vector.

    @return (float) Pearson correlation distance between the two vectors.
    """
    return 1.0 - np.abs(np.corrcoef(point1, point2)[0, 1]) #TODO:mettere i quadrati

def spearman_correlation_distance(point1, point2):
    """!
    @brief Calculate Spearman correlation distance between two vectors.

    @param[in] point1 (array_like): First vector.
    @param[in] point2 (array_like): Second vector.

    @return (float) Spearman correlation distance between the two vectors.
    """
    correlation = spearmanr(point1, point2).correlation
    return 1.0 - np.abs(correlation)
