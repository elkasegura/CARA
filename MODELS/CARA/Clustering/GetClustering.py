import os
import pandas as pd
import numpy as np
from .ClusteringTools import (group_features_by_hierarchical_clustering, params_search_hierarchical_clustering,
                             group_features_by_kmeans, params_search_kmeans,
                             group_features_by_dbscan, params_search_dbscan,
                             group_features_by_kmedoids, params_search_kmedoids)

class GetClustering():
    def __init__(self, data, output_folder, cluster_strategy, cluster_parameters, cluster_on_correlation, number_of_clusters=None, range_n_clusters=0):
        self.data = data
        self.output_folder = output_folder
        self.cluster_strategy = cluster_strategy
        self.cluster_parameters = cluster_parameters
        self.cluster_on_correlation = cluster_on_correlation
        self.number_of_clusters = number_of_clusters
        self.range_n_clusters = range_n_clusters

    def get_clusters(self):
        """
        Ottiene i cluster delle features. Se il file clusters.csv esiste già,
        lo carica. Altrimenti esegue il clustering e lo salva.
        
        :return: DataFrame contenente i cluster delle features.
        """
        # Verifica se il file clusters.csv esiste già
        clusters_file_path = os.path.join(self.output_folder, "clusters.csv")
        
        if os.path.exists(clusters_file_path):
            print(f"File clusters.csv già esistente in {self.output_folder}. Caricamento in corso...")
            clusters_df = pd.read_csv(clusters_file_path)
            return clusters_df
        
        print(f"File clusters.csv non trovato. Esecuzione del clustering con strategia: {self.cluster_strategy}")
        
        # Qui aggiungiamo il calcolo della correlazione se cluster_on_correlation è True
        if self.cluster_on_correlation:
            numeric_df = self.data.select_dtypes(include='number').copy()

            # Rimuove feature costanti (std=0) che causano NaN in corrcoef
            variances = numeric_df.var(axis=0, ddof=0)
            valid_cols = variances[variances > 0].index

            if len(valid_cols) < 2:
                raise ValueError("Feature numeriche non costanti insufficienti per calcolare la correlazione.")

            numeric_df = numeric_df[valid_cols]

            corr_matrix = np.corrcoef(numeric_df.T)
            correlation_df = pd.DataFrame(corr_matrix, index=valid_cols, columns=valid_cols)

            # Sanitizza eventuali NaN/Inf residui
            correlation_df = correlation_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            np.fill_diagonal(correlation_df.values, 1.0)

            self.data = correlation_df

        if self.cluster_strategy == 'Hierarchical':
            if self.cluster_parameters == "searching":
                self.cluster_parameters = params_search_hierarchical_clustering(self.data, output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)
                clusters_df = group_features_by_hierarchical_clustering(self.data, self.cluster_parameters['params'], mode='clustering', output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)
            else:
                clusters_df = group_features_by_hierarchical_clustering(self.data, self.cluster_parameters, mode='clustering', output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)

        elif self.cluster_strategy == 'DBScan':
            if self.cluster_parameters == "searching":
                self.cluster_parameters = params_search_dbscan(self.data, output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)
                clusters_df = group_features_by_dbscan(self.data, self.cluster_parameters['params'], mode='clustering', output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)
            else:
                clusters_df = group_features_by_dbscan(self.data, self.cluster_parameters, mode='clustering', output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)

        elif self.cluster_strategy == 'KMeans':
            if self.cluster_parameters == "searching":
                self.cluster_parameters = params_search_kmeans(self.data, output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)
                clusters_df = group_features_by_kmeans(self.data, self.cluster_parameters['params'], mode='clustering', output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)
            else:
                clusters_df = group_features_by_kmeans(self.data, self.cluster_parameters, mode='clustering', output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)

        elif self.cluster_strategy == 'KMedoids':
            if self.cluster_parameters == "searching":
                self.cluster_parameters = params_search_kmedoids(self.data, output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)
                clusters_df = group_features_by_kmedoids(self.data, self.cluster_parameters['params'], mode='clustering', output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)
            else:
                clusters_df = group_features_by_kmedoids(self.data, self.cluster_parameters, mode='clustering', output_dir=self.output_folder, number_of_clusters=self.number_of_clusters, range_n_clusters=self.range_n_clusters)

        else:
            raise ValueError(f"Strategia di clustering sconosciuta: {self.cluster_strategy}")
        
        # Validazione output clustering prima del salvataggio
        if not isinstance(clusters_df, pd.DataFrame):
            if isinstance(clusters_df, dict) and 'error' in clusters_df:
                raise RuntimeError(f"Clustering fallito: {clusters_df['error']} | dettagli: {clusters_df}")
            raise TypeError(f"Output clustering non valido: atteso DataFrame, ottenuto {type(clusters_df)}")

        required_cols = {'Feature', 'Cluster'}
        if not required_cols.issubset(set(clusters_df.columns)):
            raise ValueError(f"DataFrame cluster non valido. Colonne richieste: {required_cols}, trovate: {set(clusters_df.columns)}")

        clusters_df.to_csv(clusters_file_path, index=False)
        print(f"Clusters salvati in {clusters_file_path}")
        
        return clusters_df