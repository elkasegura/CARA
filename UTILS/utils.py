
'''def set_seed(seed: int = 42, deterministic: bool = True, disable_gpu: bool = False):
    import os, random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # TensorFlow solo si está instalado y lo necesitas
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        tf = None

    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass '''


import os
import random
import numpy as np
import torch
import tensorflow as tf

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)

    if deterministic:
        #CLAVE para CUDA >= 10.2
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Deshabilitar GPU para garantizar determinismo
        # Hbilitar configurarcion determinista con GPU,
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
