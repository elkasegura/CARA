from .utils import set_seed
set_seed(seed=42, deterministic=True)

import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
from . import unsupervised_evaluation
from ..MODELS.DRAE import DRAE
from ..MODELS.SMLAE import SMLAE
from ..MODELS.RFAE import cal 
from ..MODELS.CAE import ConcreteAutoencoderFeatureSelector
from ..MODELS.AEFS import AEFS
from ..MODELS.DGA import DGA
from ..MODELS.CARA.DataPreparation import DataPreparationFactory
from ..MODELS.CARA.tools import Config, LoggerCARA, get_scaler
from ..MODELS.CARA.Clustering import GetClustering
from ..MODELS.CARA.Filters import create_filter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import datetime
from .conf_models import _drae_config_for_num_features, _smlae_config_for_num_features, _aefs_config_for_num_features, _dga_config_for_num_features




# Para metodos MCFS, NDFS
num_features = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#Funcion para exraer los resultados con el num_features = X (todas las features)pero solo con la ACC y NMI

def resultsAllFeatures(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Evalúa con KMeans usando TODAS las features, 'repeat' repeticiones.
    Guarda CSV en: Model_results/AllFeatures/results_AllFeatures_{dataset_name}
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(float)
    y = mat["Y"][:, 0]

    num_cluster = len(np.unique(y))

    acc_runs, nmi_runs = [], []
    t0 = time.time()
    for _ in range(repeat):
        nmi, acc = unsupervised_evaluation.evaluation(
            X_selected=X,
            n_clusters=num_cluster,
            y=y
        )
        acc_runs.append(float(acc))
        nmi_runs.append(float(nmi))
    elapsed = time.time() - t0

    results = {
        "num_features": X.shape[1],
        "accuracy": float(np.mean(acc_runs)),
        "std_acc": float(np.std(acc_runs, ddof=1)),
        "nmi": float(np.mean(nmi_runs)),
        "std_nmi": float(np.std(nmi_runs, ddof=1)),
        "time_sec": float(elapsed),
    }

    df = pd.DataFrame([results])

    results_dir = Path(__file__).resolve().parents[1] / "Model_results" / "AllFeatures" / f"results_AllFeatures_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"AllFeatures_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    return df

def resultsDRAE(dataset_name: str, repeats: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre DRAE y evalúa con KMeans 'repeats' repeticiones.
    Guarda CSV en: Model_results/DRAE/results_DRAE_{dataset}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(float)   # data
    y = mat["Y"][:, 0]           # label

    XT = X.T                     # (d, n)
    n_clusters = len(np.unique(y))


    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        set_seed(42, deterministic=True)
        t0 = time.time()

        cfg = _drae_config_for_num_features(dataset_name, num_fea)

        # --- Entrenar DRAE con su config (SIN CAMBIOS) ---
        drae = DRAE(
            d_hidden=cfg["d_hidden"],
            n_clusters=n_clusters,
            alpha=1e8,
            beta=cfg["beta"],
            gamma=cfg["gamma"],
            zeta=cfg["zeta"],
            lam=cfg["lam"],
        )
        drae.fit(XT, epochs=100)

        # Seleccionar top features
        idx, scores = drae.select_top_genes(num_fea)
        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]  # (n, num_fea)

        acc_runs, nmi_runs = [], []

        # Evaluación (KMeans repeats)
        for _ in range(repeats):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y,
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": int(num_fea),
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "DRAE"
        / f"results_DRAE_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"DRAE_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"DRAE_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df



def resultsSMLAE(dataset_name: str, repeats: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre SMLAE y evalúa con KMeans 'repeats' repeticiones.
    Guarda CSV en: Model_results/SMLAE/results_SMLAE_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(float)   # data
    y = mat["Y"][:, 0]           # label

    X = normalize_zscore(X)

    XT = X.T                     # (d, n)
    d, n = XT.shape
    n_clusters1 = len(np.unique(y))   # clusters reales para evaluación
    n_clusters = np.sqrt(n / 2)    # definición usada por SMLAE

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()
        set_seed(42, deterministic=True)

        cfg = _smlae_config_for_num_features(dataset_name, num_fea)

        # --- Entrenar SMLAE con su config ---
        smale = SMLAE(
            n_clusters=n_clusters,
            d_hidden=cfg["d_hidden"],
            n_neighbors=5,
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            gamma=cfg["gamma"],
            omega=cfg["omega"],
            rho1=cfg["rho1"],
            rho2=cfg["rho2"],
        )
        smale.fit(XT, epochs=100)

        # Seleccionar top features
        idx, scores = smale.select_top_genes(num_fea)
        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]  # (n, num_fea)

        acc_runs, nmi_runs = [], []

        # Evaluación (KMeans repeats)
        for _ in range(repeats):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters1,
                y=y,
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": int(num_fea),
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "SMLAE"
        / f"results_SMLAE_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"SMLAE_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"SMLAE_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df


def resultsCAE(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre CAE y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/CAE/results_CAE_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)
    y = mat["Y"].reshape(-1)
    n_clusters = len(np.unique(y))
    d = X.shape[1]

    X = normalize_zscore(X)

    # output_function mapea (n, K) -> (n, d)  (decoder SIN CAMBIOS)
    def output_function(selected_features):
        model = Sequential([
            Dense(3*num_fea//2, activation="relu", input_shape=(selected_features.shape[1],)),
            Dense(d, activation="linear")
        ])
        return model(selected_features)

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()
        set_seed(42, deterministic=True)

        cae = ConcreteAutoencoderFeatureSelector(
            num_fea,
            output_function=output_function,
            num_epochs=200,
            batch_size=None,
            learning_rate=0.001,
            start_temp=10.0,
            min_temp=0.1,
            tryout_limit=5,
        )

        cae.fit(X)

        idx = cae.get_support(indices=True).astype(int)
        selected_features = X[:, idx]

        acc_runs, nmi_runs = [], []

        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y,
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": int(num_fea),
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "CAE"
        / f"results_CAE_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"CAE_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"CAE_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)  

    return df


def resultsRFAE(dataset_name: str, epochs: int = 1000, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre RFAE y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/RFAE/results_RFAE_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)
    y = mat["Y"].reshape(-1)
    n_clusters = len(np.unique(y))

    X = normalize_zscore(X)

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()

        # -------- RFAE training & feature selection --------
        set_seed(42, deterministic=True)
        idx = cal(
            p_data_arr=X,
            p_label_arr_onehot=y,
            p_key_feture_number=num_fea,
            datasetname=dataset_name,
            p_epochs_number=epochs,
            p_batch_size_value=128,
            clf=False,
            p_seed=0,
            device=0
        )

        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]

        acc_runs, nmi_runs = [], []

        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": num_fea,
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "RFAE"
        / f"results_RFAE_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"RFAE_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"RFAE_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df

def normalize_zscore(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / (X_std + 1e-8)  # Agregar un pequeño valor para evitar división por cero
    return X_norm

def resultsAEFS(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre AEFS y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/AEFS/results_AEFS_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    
    # AEFS settings EXACTLY as in the paper
    # - hidden ∈ {128, 256, 512, 1024}
    # - sigmoid encoder, linear decoder (handled inside AEFS)

    hidden = 256        # one valid paper value
    alpha = 1e8         # sparsity parameter (to be tuned in grid in experiments)
    beta = 0.001        # weight decay
    lr = 1e-3
    epochs = 100
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)
    y = mat["Y"].reshape(-1)
    n_clusters = len(np.unique(y))

    X = normalize_zscore(X)

    results = []
    indexfeatures_all = {}


    for num_fea in num_features:
        t0 = time.time()
        set_seed(42, deterministic=True)

        cfg = _aefs_config_for_num_features(dataset_name, num_fea)

        aefs = AEFS(hidden=cfg["hidden"], alpha=cfg["alpha"], beta=cfg["beta"], lr=cfg["lr"])

        aefs.fit(X, epochs=100)
        
        # Seleccionar top features
        idx, scores = aefs.select_top_features(num_fea)
        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]  # (n, num_fea)

        acc_runs, nmi_runs = [], []

        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": num_fea,
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "AEFS"
        / f"results_AEFS_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"AEFS_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"AEFS_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df


def resultsDGA(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre DGA y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/DGA/results_DGA_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)
    y = mat["Y"].reshape(-1)
    n_clusters = len(np.unique(y))
    n_samples, d_features = X.shape

    X = normalize_zscore(X)

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()
        set_seed(42, deterministic=True)
        cfg = _dga_config_for_num_features(dataset_name, num_fea)

        dga = DGA( 
            hidden=cfg["hidden"],
            gamma=cfg["gamma"],
            lam=cfg["lam"],
            lr=1e-3)

        dga.fit(X, epochs=100, B=n_samples)
        
        # Seleccionar top features
        idx, scores = dga.select_top_features(num_fea)
        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]  # (n, num_fea)

        acc_runs, nmi_runs = [], []

        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0
        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": num_fea,
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "DGA"
        / f"results_DGA_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"DGA_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"DGA_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df


def resultsCARA(dataset_name: str, num_features: int = 10, clustering_technique: str = "hierarchical", 
                 config_path: str = None, repeat: int = 20) -> pd.DataFrame:
    """
    Load dataset and run CARA with a specific configuration.
    Saves results to: Model_results/CARA/results_CARA_{dataset_name}
    
    Args:
        dataset_name: Name of the dataset
        num_features: Number of features to select (default: 10)
        clustering_technique: Clustering technique to use (default: "hierarchical")
        config_path: Path to YAML config file (if None, constructs default path)
        repeat: Number of evaluation repetitions
    """
    
    if config_path is None:
        # Construct config path based on parameters
        config_path = f"CARA_configs/{dataset_name}/config_{dataset_name}_{num_features}_{clustering_technique}.yaml"
    
    t0 = time.time()
    
    config = Config(config_path)
    logger = LoggerCARA(config.output.output_folder)
    logger.log_config(config)
    scaler = get_scaler(config.dataset.scaler)

    logger.log_section("DATA LOADING")
    data_preparation = DataPreparationFactory(
        dataset_name=config.dataset.name,
        input_folder=config.dataset.input_folder,
        output_folder=config.output.output_folder,
        label_column=config.dataset.label_column
    )

    data = data_preparation.export_data()

    logger.log_section("CLUSTERING")
    clustering = GetClustering(
        data=data['features_data'],
        output_folder=config.output.output_folder,
        cluster_strategy=config.clustering.strategy,
        cluster_parameters=config.clustering.parameters,
        cluster_on_correlation=config.clustering.cluster_on_correlation,
        number_of_clusters=config.clustering.number_of_clusters,
        range_n_clusters=config.clustering.range_n_clusters
    )
    clusters = clustering.get_clusters()

    features = data['features'].copy()
    features_data = data['features_data'][features].copy()
    iteration = 0

    logger.log_section("FILTERING")
    FeatureFilter = create_filter(config.filters, iteration, config.output.output_folder, scaler)
    selected_features = FeatureFilter.filter(features_data, clusters)
    logger.log_message(f"Selected features: {selected_features}")

    logger.log_section("EVALUATION")
    selected_features_data = data['features_data'][selected_features].copy()
    labels = data['labels'].copy().values

    nmi_scores, acc_scores = [], []

    for _ in range(repeat):
        nmi, acc = unsupervised_evaluation.evaluation(
            X_selected=selected_features_data,
            n_clusters=len(np.unique(labels)),
            y=labels
        )
        nmi_scores.append(float(nmi))
        acc_scores.append(float(acc))

    elapsed = time.time() - t0

    results = {
        "num_features": len(selected_features),
        "accuracy": float(np.mean(acc_scores)),
        "std_acc": float(np.std(acc_scores, ddof=1)),
        "nmi": float(np.mean(nmi_scores)),
        "std_nmi": float(np.std(nmi_scores, ddof=1)),
        "time_sec": float(elapsed),
    }

    df = pd.DataFrame([results])

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "CARA"
        / f"results_CARA_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"CARA_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    logger.log_message(f"Results saved to: {out_csv}")

    return df
