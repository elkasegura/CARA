import pandas as pd
import numpy as np
from scipy.spatial import distance
from .loggers import FilterLogger  # Assuming you have a logger class

class MedoidFilter:
    def __init__(self, iteration, output_dir, scaler, distance_metric='euclidean'):
        """
        Initializes the Medoid Filter.

        Args:
            iteration (int): Current iteration number.
            output_dir (str): Output directory for logs.
        """
        self.iteration = iteration
        self.output_dir = output_dir
        self.scaler = scaler
        self.distance_metric = distance_metric
        self.logger = FilterLogger(iteration, output_dir=output_dir)

    def get_data(self, data, clusters, i_clust):
        """
        Selects data for the current cluster.

        Args:
            data (pd.DataFrame): Input data.
            clusters (pd.DataFrame): Cluster assignments.
            i_clust (int): Current cluster ID.

        Returns:
            tuple: (feature names, data for the cluster)
        """
        # Select features for the current cluster
        current_features = clusters['Feature'][clusters['Cluster'] == i_clust].values
        X = data[current_features].T.copy()

        if X.shape[0] == 0 or X.shape[1] == 0:
            self.logger.log_message(f"Skipping medoid selection for cluster {i_clust} due to empty feature set.")
            return [], None
        
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), index=X.index)

        return current_features, X_scaled

    def calculate_distances(self, data):
        """
        Calculates the distances between each data point using scipy.

        Args:
            data (np.ndarray): Input data as a numpy array.

        Returns:
            np.ndarray: Distance matrix.
        """
        # Convert DataFrame to numpy array if it's not already
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        
        distances = distance.cdist(data, data, self.distance_metric)
        return distances

    def select_medoid(self, data, feature_names):
        """
        Selects the medoid feature (the feature with the minimum sum of distances to all other features).

        Args:
            data (pd.DataFrame): Input data.
            feature_names (list): List of feature names.

        Returns:
            str: Name of the selected medoid feature.
        """
        distances = self.calculate_distances(data)
        sum_distances = np.sum(distances, axis=1)
        medoid_index = np.argmin(sum_distances)
        selected_gene = feature_names[medoid_index]
        return selected_gene

    def filter(self, data, clusters):
        """
        Filters features by selecting the medoid feature for each cluster.

        Args:
            data (pd.DataFrame): Input data.
            clusters (pd.DataFrame): Cluster assignments.

        Returns:
            list: List of selected medoid features.
        """
        selection = []
        n_clusters = max(clusters['Cluster']) + 1

        self.logger.log_message(f'Starting medoid filtering for {n_clusters - 1} clusters')

        for i in range(0, n_clusters):
            feature_names, cluster_data = self.get_data(data, clusters, i)

            print(f'Cluster {i}: {len(feature_names)} features')
            print(f'Cluster {i} data shape: {cluster_data.shape if cluster_data is not None else "N/A"}')
            
            if cluster_data is None or len(feature_names) == 0:
                continue

            selected_gene = self.select_medoid(cluster_data, feature_names)
            selection.append(selected_gene)

            self.logger.log_message(f'Filtered Cluster {i}/{n_clusters - 1}, selected gene: {selected_gene}')

        # Log final summary
        self.logger.log_selected_features_summary(selection)
        self.logger.log_completion()

        return selection