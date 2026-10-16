from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
import os
from joblib import Parallel, delayed
from .tools import ClusteringLogger, evaluate_cluster, compute_feature_distances, select_best_clustering
import matplotlib.pyplot as plt


def group_features_by_hierarchical_clustering(df, params_hierarchical_clustering, mode='searching', output_dir='dataset_training', number_of_clusters=None, range_n_clusters=0):
    """Esegue il clustering gerarchico e valuta i risultati con diverse metriche"""
    
    distance_metric = params_hierarchical_clustering['distance_metric']
    linkage_method = params_hierarchical_clustering['linkage_method']
    
    logger = ClusteringLogger(log_filename=f"hierarchical_clustering_{distance_metric}_{linkage_method}.log", log_dir=os.path.join(output_dir,'clustering_searh_logs'))
    
    try:
        # Calcola la matrice delle distanze tra le feature
        distance_matrix = compute_feature_distances(df, distance_metric)
        # Esegui il clustering gerarchico
        Z = linkage(distance_matrix, method=linkage_method)
    except Exception as e:
        logger.log(f"ERRORE nel calcolo del linkage: {str(e)}")
        logger.log(f"Parametri: distance_metric={distance_metric}, linkage_method={linkage_method}")
        
        # Ritorna risultati con metriche pessime per indicare il fallimento
        bad_result = {
            'num_clusters': number_of_clusters,
            'silhouette_score': -1.0,  # Silhouette peggiore possibile
            'davies_bouldin_score': 999999.0,  # Davies-Bouldin pessimo
            'avg_features_per_cluster': 0.0,
            'single_feature_clusters_percentage': 100.0  # Tutti cluster singoli (pessimo)
        }
        
        return {
            'params': params_hierarchical_clustering,
            'best_result': bad_result,
            'error': str(e)
        }
    
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
        logger.log(f'Numero di features: {num_features}, Numero di pazienti: {num_patients}')
        logger.log(f'Numero minimo di features per cluster: {min_features_per_cluster}, Numero massimo di features per cluster: {max_features_per_cluster}')

        # Numero massimo e minimo di cluster derivati dall'euristica
        max_clusters = max(2, num_features // min_features_per_cluster)
        min_clusters = max(2, num_features // max_features_per_cluster)
    logger.log(f'Numero minimo di cluster: {min_clusters}, Numero massimo di cluster: {max_clusters}')

    dict_results_list = []
    # Cerca il miglior numero di cluster per entrambi gli score
    for num_clusters in range(min_clusters, max_clusters + 1):
        try:
            clusters = fcluster(Z, num_clusters, criterion='maxclust')
            
            score_silhouette, avg_features_per_cluster, single_feature_clusters = evaluate_cluster(df, clusters, 'silhouette', min_features_per_cluster)
            single_feature_clusters_percentage = single_feature_clusters / num_clusters * 100
            score_dbs, _, _ = evaluate_cluster(df, clusters, 'davies-bouldin', min_features_per_cluster)
            logger.log(f"Num clusters: {num_clusters}, Silhouette Score: {score_silhouette:.4f}, Davies-Bouldin Score: {score_dbs:.4f}, Avg features per cluster: {avg_features_per_cluster:.2f}, Single feature clusters %: {single_feature_clusters_percentage}")
            dict_results = {'num_clusters': num_clusters, 'silhouette_score': score_silhouette, 'davies_bouldin_score': score_dbs, 'avg_features_per_cluster': avg_features_per_cluster, 'single_feature_clusters_percentage': single_feature_clusters_percentage}
            dict_results_list.append(dict_results)
        except Exception as e:
            logger.log(f"⚠ ERRORE nel clustering con {num_clusters} cluster: {str(e)}")
            continue
    
    if not dict_results_list:
        logger.log("⚠ ERRORE: Nessun risultato valido ottenuto nel clustering")
        bad_result = {
            'num_clusters': number_of_clusters if number_of_clusters else 2,
            'silhouette_score': -1.0,
            'davies_bouldin_score': 999999.0,
            'avg_features_per_cluster': 0.0,
            'single_feature_clusters_percentage': 100.0
        }
        return {
            'params': params_hierarchical_clustering,
            'best_result': bad_result,
            'error': 'No valid clustering results'
        }
    
    best_result = select_best_clustering(dict_results_list)
    logger.log(f'Best result: {best_result}')
    
    if mode == 'searching':
        return {
            'params': params_hierarchical_clustering,
            'best_result': best_result
        }
    
    final_logger = ClusteringLogger(log_filename=f"hierarchical_clustering_best.log", log_dir=output_dir)
    final_logger.log(f"Numero di cluster: {best_result['num_clusters']}")
    final_logger.log(f"Silhouette Score: {best_result['silhouette_score']:.4f}")
    final_logger.log(f"Davies-Bouldin Score: {best_result['davies_bouldin_score']:.4f}")
    final_logger.log(f"Numero medio di features per cluster: {best_result['avg_features_per_cluster']:.2f}")
    final_logger.log(f"Numero di cluster con un solo punto (%): {best_result['single_feature_clusters_percentage']:.2f}")
  
    clusters = fcluster(Z, best_result['num_clusters'], criterion='maxclust')
    
    # Crea un dizionario che mappa ogni feature al suo cluster
    feature_to_cluster = {feature: cluster for feature, cluster in zip(df.columns, clusters)}
    
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
        
def params_search_hierarchical_clustering(df, output_dir='dataset_training', number_of_clusters=None, range_n_clusters=0):
    """Esegue la ricerca dei migliori parametri per Hierarchical Clustering"""
    
    metric_feature_distance = ['euclidean', 'pearson', 'cosine', 'spearman']
    #metric_feature_distance = ['euclidean']
    method_linkage = ['ward', 'single', 'complete', 'average']
    logger = ClusteringLogger(log_filename="hierarchical_clustering_search.log", log_dir=os.path.join(output_dir,'clustering_searh_logs'))
    
    # Crea una lista di dizionari con tutte le combinazioni possibili di parametri
    params = [{'distance_metric': metric, 'linkage_method': linkage}
              for metric in metric_feature_distance
              for linkage in method_linkage]
    
    # Esegui la ricerca in parallelo
    jobs = [delayed(group_features_by_hierarchical_clustering)(df, p, mode = 'searching', output_dir = output_dir, number_of_clusters=number_of_clusters, range_n_clusters=range_n_clusters) for p in params]
    results = Parallel(n_jobs=-1, verbose=1)(jobs)
    
    # Filtra risultati con errori e logga
    valid_results = []
    for result in results:
        logger.log(result)
        if 'error' not in result:
            valid_results.append(result)
        else:
            logger.log(f"⚠ Configurazione fallita: {result['params']} - Errore: {result.get('error', 'Unknown')}")

    # Se non ci sono risultati validi, usa tutti i risultati (anche quelli con errori)
    if not valid_results:
        logger.log("⚠ ATTENZIONE: Nessuna configurazione valida trovata. Usando il miglior risultato disponibile tra quelli falliti.")
        valid_results = results
    
    best_configuration = min(valid_results, key=lambda x: x['best_result']['single_feature_clusters_percentage'])

    logger.log("\n### CONFIGURAZIONE FINALE ###")
    logger.log(best_configuration)

    return best_configuration