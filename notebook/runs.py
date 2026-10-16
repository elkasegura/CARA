import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
print("Added to sys.path:", PROJECT_ROOT)


from ..UTILS import results as rs
from ..UTILS.utils import set_seed


set_seed(seed=42, deterministic=True)




datasets = [
    "lymphoma",
    "leukemia",
    "GLI-85",
    "BASEHOCK", 
    "COIL20",
    "GLIOMA",
    "Isolet",
    "lung_small",
    "madelon",
    "orlraws10P",
    "PCMAC",
    "pixraw10P",
    "RELATHE",
    "SMK-CAN-187",
    "USPS",
    
]
for dataset in datasets:
    print(f"\n===== RFAE | Dataset: {dataset} =====")
    rs.resultsRFAE(dataset_name=dataset)
    print(f"\n===== DRAE | Dataset: {dataset} =====")
    rs.resultsDRAE(dataset_name=dataset)
    print(f"\n===== SMLAE | Dataset: {dataset} =====")
    rs.resultsSMLAE(dataset_name=dataset)
    print(f"\n===== CAE | Dataset: {dataset} =====")
    rs.resultsCAE(dataset_name=dataset)
    print(f"\n===== AEFS | Dataset: {dataset} =====")
    rs.resultsAEFS(dataset_name=dataset)
    print(f"\n===== DGA | Dataset: {dataset} =====")
    rs.resultsDGA(dataset_name=dataset)
    
    # CARA with multiple configurations
    num_features_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    clustering_techniques = ["hierarchical", "kmeans"]
    
    for num_fea in num_features_list:
        for clustering_tech in clustering_techniques:
            print(f"\n===== CARA | Dataset: {dataset} | Features: {num_fea} | Clustering: {clustering_tech} =====")
            rs.resultsCARA(dataset_name=dataset, num_features=num_fea, clustering_technique=clustering_tech)
    
    print(f"\n===== ALL | Dataset: {dataset} =====")
    rs.resultsAllFeatures(dataset_name=dataset)