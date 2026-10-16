from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from .tools import ClusteringLogger, evaluate_cluster, remap_labels_distinct_noise
import pandas as pd
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt

def group_features_by_dbscan(df, params_dbscan, mode='searching', output_dir='dataset_training', number_of_clusters=None, range_n_clusters=0):
    """Esegue il clustering DBSCAN e valuta i risultati con diverse metriche"""
    
    eps = params_dbscan['eps']
    min_samples = params_dbscan['min_samples']
    metric_distance = params_dbscan['metric_distance']
    
    logger = ClusteringLogger(log_filename=f"dbscan_clustering_{eps}_{min_samples}_{metric_distance}.log", log_dir=os.path.join(output_dir,'clustering_searh_logs'))
    
    # Standardizza i dati
    numeric_df = df.select_dtypes(include='number')
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(numeric_df.T)
    
    # Definizione dei limiti per il numero di cluster
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
    
    # Esegui il clustering DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric = metric_distance).fit(scaled_df)
    labels = remap_labels_distinct_noise(db.labels_)
    
    num_clusters = len(set(labels))
    # Valuta il clustering
    score_silhouette, avg_features_per_cluster, single_feature_clusters = evaluate_cluster(df, labels, 'silhouette', min_features_per_cluster)
    score_dbs, _, _ = evaluate_cluster(df, labels, 'davies-bouldin', min_features_per_cluster)
    single_feature_clusters_percentage = single_feature_clusters / num_clusters * 100
    logger.log(f"Num clusters: {num_clusters}, Silhouette Score: {score_silhouette:.4f}, Davies-Bouldin Score: {score_dbs:.4f}, Avg features per cluster: {avg_features_per_cluster:.2f}, Single feature clusters %: {single_feature_clusters_percentage}")
    best_result = {'num_clusters': num_clusters, 'silhouette_score': score_silhouette, 'davies_bouldin_score': score_dbs, 'avg_features_per_cluster': avg_features_per_cluster, 'single_feature_clusters_percentage': single_feature_clusters_percentage}

    if mode == 'searching':
        return {
            'params': params_dbscan,
            'best_result': best_result
        }
    
    final_logger = ClusteringLogger(log_filename=f"dbscan_clustering_best.log", log_dir=output_dir)
    final_logger.log(f"Numero di cluster: {best_result['num_clusters']}")
    final_logger.log(f"Silhouette Score: {best_result['silhouette_score']:.4f}")
    final_logger.log(f"Davies-Bouldin Score: {best_result['davies_bouldin_score']:.4f}")
    final_logger.log(f"Numero medio di features per cluster: {best_result['avg_features_per_cluster']:.2f}")
    final_logger.log(f"Numero di cluster con un solo punto (%): {best_result['single_feature_clusters_percentage']:.2f}")
  
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

def params_search_dbscan(df, output_dir='dataset_training', number_of_clusters=None, range_n_clusters=0):
    """Esegue la ricerca dei migliori parametri per DBSCAN"""
    
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    min_samples_values = [5, 10, 15, 20]
    metric_distance = ['euclidean', 'correlation', 'cosine']
    logger = ClusteringLogger(log_filename="dbscan_clustering_search.log", log_dir=os.path.join(output_dir,'clustering_searh_logs'))
    
    # Crea una lista di dizionari con tutte le combinazioni possibili di parametri
    params = [{'eps': eps, 'min_samples': min_samples, 'metric_distance': metric}
              for eps in eps_values
              for min_samples in min_samples_values
              for metric in metric_distance]
    
    # Esegui la ricerca in parallelo
    jobs = [delayed(group_features_by_dbscan)(df, p, mode='searching', output_dir=output_dir, number_of_clusters=number_of_clusters, range_n_clusters=range_n_clusters) for p in params]
    results = Parallel(n_jobs=-1, verbose=1)(jobs)
    
    for result in results:
        logger.log(result)
    
    best_configuration = min(results, key=lambda x: x['best_result']['single_feature_clusters_percentage'])
    
    logger.log("\n### CONFIGURAZIONE FINALE ###")
    logger.log(best_configuration)

    return best_configuration