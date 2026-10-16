import torch
import torch.nn as nn

class LinearAE(nn.Module):
    def __init__(self, n_input, n_layers=3):
        super(LinearAE, self).__init__()
        self.n_input = n_input
        self.n_layers = max(1, int(n_layers))

        hidden_sizes = self._build_hidden_sizes(n_input, self.n_layers)
        layer_sizes = [n_input] + hidden_sizes + list(reversed(hidden_sizes)) + [n_input]

        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.relu = nn.ReLU()

        # Inizializzazione Xavier
        self._initialize_weights()

    def _build_hidden_sizes(self, n_input, n_layers):
        hidden_sizes = []
        current_size = max(1, n_input // 2)

        for _ in range(n_layers):
            hidden_sizes.append(max(1, current_size))
            current_size = max(1, current_size // 2)

        return hidden_sizes

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))

        x = self.layers[-1](x)
        
        return x