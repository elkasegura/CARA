from ..MODELS.CARA.DataPreparation import DataPreparationFactory
from ..MODELS.CARA.tools import Config, LoggerCARA, get_scaler
from ..MODELS.CARA.Clustering import GetClustering
from ..MODELS.CARA.Filters import create_filter
from ..UTILS import unsupervised_evaluation
import numpy as np
import os
import pandas as pd
import argparse





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run CARA with a specific configuration.")
    parser.add_argument("--config", type=str, default="CARA_configs/basehock/config_basehock_10_hierarchical.yaml", 
                        help="Path to the YAML configuration file (default: CARA_configs/example_config.yaml)")
    args = parser.parse_args()
    
    config_path = args.config
    
    config = Config(config_path)
    logger = LoggerCARA(config.output.output_folder)
    logger.log_config(config)
    scaler = get_scaler(config.dataset.scaler)

    logger.log_section("DATA LOADING")
    data_preparation = DataPreparationFactory(dataset_name=config.dataset.name,
                                              input_folder=config.dataset.input_folder,
                                              output_folder=config.output.output_folder,
                                              label_column=config.dataset.label_column)

    data = data_preparation.export_data()

    logger.log_section("CLUSTERING")

    clustering = GetClustering(data = data['features_data'], 
                               output_folder=config.output.output_folder, 
                               cluster_strategy=config.clustering.strategy, 
                               cluster_parameters=config.clustering.parameters, 
                               cluster_on_correlation=config.clustering.cluster_on_correlation,
                               number_of_clusters=config.clustering.number_of_clusters,
                               range_n_clusters=config.clustering.range_n_clusters)

    clusters = clustering.get_clusters()


    features = data['features'].copy()
    features_data = data['features_data'][features].copy()
    iteration = 0

    logger.log_section("FILTERING")

    FeatureFilter = create_filter(config.filters, iteration, config.output.output_folder, scaler)
    
    print(f'feature data shape before filtering: {features_data.shape}')
    print(f'clusters shape: {clusters.shape}')
            
    selected_features = FeatureFilter.filter(features_data, clusters)
    
    logger.log_message(f"Selected features: {selected_features}")

    logger.log_section("EVALUATION")
    
    selected_features_data = data['features_data'][selected_features].copy()
    print(f'selected features data shape: {selected_features_data.shape}')
    labels = data['labels'].copy().values

    repeats = 20

    nmi_scores, acc_scores = [], []

    for _ in range(repeats):
        nmi, acc = unsupervised_evaluation.evaluation(
            X_selected=selected_features_data,
            n_clusters=len(np.unique(labels)),
            y=labels
        )
        nmi_scores.append(nmi)
        acc_scores.append(acc)

    std_nmi = np.std(nmi_scores, ddof=1)
    std_acc = np.std(acc_scores, ddof=1)

    logger.log_message(f"NMI: {np.mean(nmi_scores):.4f} ± {std_nmi:.4f}")
    logger.log_message(f"ACC: {np.mean(acc_scores):.4f} ± {std_acc:.4f}")