from .AEFilter import AEFilter
from .MedoidFilter import MedoidFilter

def create_filter(filter_config, iteration, output_dir, scaler):
    """
    Crea un'istanza del filtro specificato nella configurazione.

    Args:
        config: Oggetto di configurazione contenente le impostazioni del filtro.
        iteration (int): Numero dell'iterazione corrente.
        output_dir (str): Directory di output per i risultati.

    Returns:
        Un'istanza della classe filtro appropriata.
    """
    filter_type = filter_config.filter_type

    if filter_type == 'AE':
        ae_config = filter_config.autoencoder_parameters
        return AEFilter(
            lr=ae_config.lr,
            epoch=ae_config.epochs,
            model_name=ae_config.AE_name,
            iteration=iteration,
            output_dir=output_dir,
            scaler=scaler,
            n_layers=ae_config.n_layers,
        )
    elif filter_type == 'Medoid':
        return MedoidFilter(
            iteration=iteration,
            output_dir=output_dir,
            distance_metric=filter_config.medoid_parameters.distance_metric,
            scaler=scaler,
        )
    else:
        raise ValueError(f"Tipo di filtro non valido: {filter_type}. I tipi validi sono 'AE' e 'Medoid'.")