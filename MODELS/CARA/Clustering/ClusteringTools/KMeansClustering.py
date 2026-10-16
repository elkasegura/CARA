from ..pyclustering_custom.cluster.kmeans import kmeans
from ..pyclustering_custom.cluster.center_initializer import kmeans_plusplus_initializer
from ..pyclustering_custom.utils.metric import distance_metric, type_metric
from sklearn.preprocessing import StandardScaler
from .tools import ClusteringLogger, evaluate_cluster, select_best_clustering, pearson_correlation_distance, spearman_correlation_distance
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import os
import matplotlib.pyplot as plt

def group_features_by_kmeans(df, params_kmeans, mode='searching', output_dir='dataset_training', number_of_clusters=None, range_n_clusters=0):
    """Esegue il clustering KMeans e valuta i risultati con diverse metriche"""
    
    num_features = df.shape[1]
    num_patients = df.shape[0]

    if number_of_clusters is not None:
        # Se il numero di cluster è specificato, usalo direttamente
        min_clusters = max(2, number_of_clusters - range_n_clusters)
        max_clusters = number_of_clusters + range_n_clusters
        min_features_per_cluster = max(2, num_features // max_clusters)
        max_features_per_cluster = max(2, num_features // min_clusters)
    else:
    # Euristica per il numero minimo e massimo di feature per cluster
        min_features_per_cluster = max(2, num_patients * 2)
        max_features_per_cluster = max(2, num_patients * 10)

        # Numero massimo e minimo di cluster derivati dall'euristica
        max_clusters = max(2, num_features // min_features_per_cluster)
        min_clusters = max(2, num_features // max_features_per_cluster)

    metric_type = params_kmeans['metric_type']
    
    if metric_type == 'euclidean':
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    logger = ClusteringLogger(log_filename=f"kmeans_clustering_{metric_type}.log", log_dir=os.path.join(output_dir,'clustering_searh_logs'))
    
    logger.log(f'Numero di features: {num_features}, Numero di pazienti: {num_patients}')
    logger.log(f'Numero minimo di features per cluster: {min_features_per_cluster}, Numero massimo di features per cluster: {max_features_per_cluster}')
    logger.log(f'Numero minimo di cluster: {min_clusters}, Numero massimo di cluster: {max_clusters}')

    dict_results_list = []
    
    # Esegui il clustering KMeans
    metric = distance_metric(getattr(type_metric, metric_type.upper()))
    
    patience = 10  # Numero di iterazioni consecutive con lo stesso miglior risultato
    best_result = None
    patience_counter = 0

    for iter, num_clusters in enumerate(range(min_clusters, max_clusters + 1)):
        # Inizializza i centri dei cluster
        initial_centers = kmeans_plusplus_initializer(df.T, num_clusters, random_state=42).initialize()
        
        kmeans_instance = kmeans(df.T, initial_centers, metric=metric)
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()

        # Crea le etichette dei cluster
        labels = np.zeros(num_features, dtype=int)

        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id

        # Valuta il clustering
        score_silhouette, avg_features_per_cluster, single_feature_clusters = evaluate_cluster(df, labels, 'silhouette', min_features_per_cluster)
        single_feature_clusters_percentage = single_feature_clusters / num_clusters * 100
        score_dbs, _, _ = evaluate_cluster(df, labels, 'davies-bouldin', min_features_per_cluster)
        logger.log(f"Num clusters: {num_clusters}, Silhouette Score: {score_silhouette:.4f}, Davies-Bouldin Score: {score_dbs:.4f}, Avg features per cluster: {avg_features_per_cluster:.2f}, Single feature clusters %: {single_feature_clusters_percentage}")
        dict_results = {'num_clusters': num_clusters, 'silhouette_score': score_silhouette, 'davies_bouldin_score': score_dbs, 'avg_features_per_cluster': avg_features_per_cluster, 'single_feature_clusters_percentage': single_feature_clusters_percentage}
        dict_results_list.append(dict_results)
        
        # Calcola il miglior risultato corrente
        current_best = select_best_clustering(dict_results_list)
        
        # Confronta con il miglior risultato precedente
        if best_result is None:
            best_result = current_best
            patience_counter = 0
        elif current_best['num_clusters'] == best_result['num_clusters']:
            patience_counter += 1
            logger.log(f"Il miglior risultato rimane stabile: {patience_counter}/{patience}")
            
            # Interrompi la ricerca se la pazienza è esaurita
            if patience_counter >= patience:
                logger.log(f"Early stopping attivato dopo {patience} iterazioni con lo stesso miglior risultato (num_clusters: {best_result['num_clusters']})")
                break
        else:
            best_result = current_best
            patience_counter = 0
        
    best_result = select_best_clustering(dict_results_list)
    logger.log(f'Best result: {best_result}')
    
    if mode == 'searching':
        return {
            'params': params_kmeans,
            'best_result': best_result
        }
    
    final_logger = ClusteringLogger(log_filename=f"kmeans_clustering_best.log", log_dir=output_dir)
    final_logger.log(f"Numero di cluster: {best_result['num_clusters']}")
    final_logger.log(f"Silhouette Score: {best_result['silhouette_score']:.4f}")
    final_logger.log(f"Davies-Bouldin Score: {best_result['davies_bouldin_score']:.4f}")
    final_logger.log(f"Numero medio di features per cluster: {best_result['avg_features_per_cluster']:.2f}")
    final_logger.log(f"Numero di cluster con un solo punto (%): {best_result['single_feature_clusters_percentage']:.2f}")
  
    
    # Esegui il clustering con il miglior num_clusters
    n_clusters = best_result['num_clusters']
    initial_centers = kmeans_plusplus_initializer(df.T, n_clusters, random_state=42).initialize()
    kmeans_instance = kmeans(df.T, initial_centers, metric=metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()

    # Crea le etichette dei cluster
    labels = np.zeros(num_features, dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id

    # Crea un dizionario che mappa ogni feature al suo cluster
    feature_to_cluster = {feature: cluster for feature, cluster in zip(df.columns, labels)}

    # Converte il dizionario in un DataFrame
    final_df = pd.DataFrame(list(feature_to_cluster.items()), columns=['Feature', 'Cluster'])

    # Ordina il DataFrame per cluster
    final_df = final_df.sort_values(by='Cluster').reset_index(drop=True)
    
        # Plot dei conteggi delle feature per cluster
    cluster_counts = final_df['Cluster'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar')
    plt.xlabel('Cluster')
    plt.ylabel('Numero di Features')
    plt.title('Conteggio delle Features per Cluster')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_counts_per_cluster.png'))

    return final_df

def params_search_kmeans(df, output_dir='dataset_training', number_of_clusters=None, range_n_clusters=0):
    """Esegue la ricerca dei migliori parametri per KMeans"""
    
    metric_types = ['euclidean', 'cosine', 'pearson', 'spearman']
    #metric_types = ['euclidean']
    
    logger = ClusteringLogger(log_filename="kmeans_clustering_search.log", log_dir=os.path.join(output_dir,'clustering_searh_logs'))
    
    # Crea una lista di dizionari con tutte le combinazioni possibili di parametri
    params = [{'metric_type': metric} for metric in metric_types]
    
    # Esegui la ricerca in parallelo
    jobs = [delayed(group_features_by_kmeans)(df, p, mode='searching', output_dir=output_dir, number_of_clusters=number_of_clusters, range_n_clusters=range_n_clusters) for p in params]
    results = Parallel(n_jobs=-1, verbose=1)(jobs)
    
    for result in results:
        logger.log(result)
    # Trova la combinazione di parametri con il miglior Silhouette Score
    
    best_configuration = min(results, key=lambda x: x['best_result']['single_feature_clusters_percentage'])

    logger.log("\n### CONFIGURAZIONE FINALE ###")
    logger.log(best_configuration)

    return best_configuration