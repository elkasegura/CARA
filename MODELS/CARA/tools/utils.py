from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_scaler(scaler_name: str):
    if scaler_name.lower() == "standard":
        return StandardScaler()
    elif scaler_name.lower() == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Scaler '{scaler_name}' not supported.")