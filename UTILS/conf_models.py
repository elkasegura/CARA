

#DICT con hiperparámetros para DRAE según número de features


def _drae_config_for_num_features(dataset: str, num_fea: int):
    """
    Devuelve dict de hiperparámetros DRAE según num_fea.
    """
    if dataset.lower() in ["allaml","prostate-ge", "colon", "gli-85", "leukemia", "gli-85", "tox-171" ,  "arcene", "basehock", "carcinom", "cll-sub-111", "coil20", "gisette", "glioma", "isolet", "leukemia", "lung_small", "lung", "lymphoma", "madelon", "nci9", "orl", "orlraws10p", "pcmac", "pixraw10p", "relathe", "smk-can-187", "usps", "warpar10p", "warppie10p", "yale"]:
        if num_fea in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            return dict(d_hidden=10, alpha=1e8, beta=1e2, gamma=1e5, zeta=0.001, lam=1e-1)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")

#DICT con hiperparámetros para SMLAE según número de features       

def _smlae_config_for_num_features(dataset:str, num_fea: int):
    """
    Devuelve dict de hiperparámetros SMLAE según num_fea
    """
    if dataset.lower() in ["allaml", "prostate-ge", "colon", "gli-85", "leukemia", "gli-85", "tox-171" ,  "arcene", "basehock", "carcinom", "cll-sub-111", "coil20", "gisette", "glioma", "isolet", "leukemia", "lung_small", "lung", "lymphoma", "madelon", "nci9", "orl", "orlraws10p", "pcmac", "pixraw10p", "relathe", "smk-can-187", "usps", "warpar10p", "warppie10p", "yale"]:
        if num_fea in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            return dict(d_hidden=10, n_neighbors=5, alpha=1e5, beta=0.01, gamma=1e-6, omega=1e-6, rho1=1e-5, rho2=1e-5)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
        
#DICT con hiperparámetros para AEFS según número de features
    
def _aefs_config_for_num_features(dataset: str, num_fea: int):
    """
    Devuelve dict de hiperparámetros AEFS según num_fea
    """
    if dataset.lower() in ["allaml", "prostate-ge", "colon", "gli-85", "leukemia", "gli-85", "tox-171" ,  "arcene", "basehock", "carcinom", "cll-sub-111", "coil20", "gisette", "glioma", "isolet", "leukemia", "lung_small", "lung", "lymphoma", "madelon", "nci9", "orl", "orlraws10p", "pcmac", "pixraw10p", "relathe", "smk-can-187", "usps", "warpar10p", "warppie10p", "yale"]:
        if num_fea in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            return dict(hidden=256, alpha=0.5, beta=1, lr=1e-3)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
        
#DICT con hiperparámetros para DGA según número de features
    
def _dga_config_for_num_features(dataset: str, num_fea: int):
    """
    Devuelve dict de hiperparámetros DGA según num_fea
    """
    if dataset.lower() in ["allaml", "prostate-ge", "colon", "gli-85", "leukemia", "gli-85", "tox-171" ,  "arcene", "basehock", "carcinom", "cll-sub-111", "coil20", "gisette", "glioma", "isolet", "leukemia", "lung_small", "lung", "lymphoma", "madelon", "nci9", "orl", "orlraws10p", "pcmac", "pixraw10p", "relathe", "smk-can-187", "usps", "warpar10p", "warppie10p", "yale"]:
        if num_fea in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            return dict(hidden=20, gamma=0.1, lam=0.1)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")