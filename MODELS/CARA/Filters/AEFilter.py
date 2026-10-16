import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import copy
import random
from .autoencoders import LinearAE
from .loggers import FilterLogger

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

class AEFilter:
    def __init__(self, lr, epoch, model_name, iteration, output_dir, scaler, n_layers=None):
        self.model_name = model_name
        self.model = None
        self.lr = lr
        self.epoch = epoch
        self.iteration = iteration
        self.best_weights = None
        self.best_n_layers = None
        self.output_dir = output_dir
        self.logger = FilterLogger(iteration, output_dir=output_dir)
        self.scaler = scaler
        self.n_layers = n_layers if n_layers is not None else [1, 2, 3, 4]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_data(self, data, clusters, i_clust):
        # Seleziona i geni del cluster corrente
        current_features = clusters['Feature'][clusters['Cluster'] == i_clust].values
        X = data[current_features].T.copy()
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            self.logger.log_message(f"Skipping training for cluster {i_clust} due to empty feature set.")
            X = torch.empty(0, 0)
            return X, X, X
        
        torch.manual_seed(SEED)
        random.seed(SEED)
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), index=X.index)
        # Converte i dati in tensori
        X_tensor = torch.tensor(X_scaled.to_numpy(), dtype=torch.float32)
        X_tensor = X_tensor.to(self.device)
        
        return X_scaled.shape[1], X_tensor, X_scaled

    def create_model(self, input_size, n_layers):
        if self.model_name == "LinearAE":
            model = LinearAE(input_size, n_layers=n_layers)
            model.to(self.device)
        else:
            raise ValueError(f"Modello {self.model_name} non supportato.")
        return model

    def train_model(self, model, X_tensor):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=max(self.lr * 1e-3, 1e-6),
        )
        loss_fn = nn.L1Loss()
        best_acc = float('inf')
        best_weights = None

        for epoch in range(self.epoch):
            model.train()
            optimizer.zero_grad()

            outputs = model(X_tensor)
            loss = loss_fn(outputs, X_tensor)
            loss.backward()
            optimizer.step()

            scheduler.step(loss.item())

            accuracy = torch.mean(torch.abs(outputs - X_tensor)).item()
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.log_training_progress(epoch, self.epoch, loss.item(), accuracy, current_lr)

            if accuracy < best_acc:
                best_acc = accuracy
                best_weights = copy.deepcopy(model.state_dict())

        if best_weights is not None:
            model.load_state_dict(best_weights)

        return model, best_acc

    def fit(self, data, clusters, i_clust):
        # Prepara i dati e il modello
        num_feat, X_tensor, df_scaled = self.get_data(data, clusters, i_clust)
        
        print(f'x_tensor shape: {X_tensor.shape}')
        
        if X_tensor.shape[0] == 0 or X_tensor.shape[1] == 0:
            self.logger.log_message(f"Skipping training for cluster {i_clust} due to empty feature set.")
            return None
        
        self.logger.log_cluster_start(i_clust, num_feat)

        candidate_layers = self.n_layers if isinstance(self.n_layers, list) else [self.n_layers]
        best_model = None
        best_acc = float('inf')

        for n_layers in candidate_layers:
            torch.manual_seed(SEED)
            random.seed(SEED)
            model = self.create_model(num_feat, n_layers)
            model, acc = self.train_model(model, X_tensor)

            if acc < best_acc:
                best_acc = acc
                best_model = model
                self.best_n_layers = n_layers

        self.model = best_model
        self.best_weights = copy.deepcopy(self.model.state_dict()) if self.model is not None else None

        # Previsione e selezione del miglior gene
        preds = self.predict(X_tensor)
        selected_idx = torch.argmin(torch.mean(torch.abs(X_tensor - preds), dim=1)).item()
        selected_gene = df_scaled.index.values[selected_idx]

        # Calcola l'errore di ricostruzione per il gene selezionato
        selected_error = torch.mean(torch.abs(X_tensor[selected_idx] - preds[selected_idx])).item()
        
        # Log della feature selezionata
        self.logger.log_message(f"Best n_layers for cluster {i_clust}: {self.best_n_layers}")
        self.logger.log_feature_selection(i_clust, selected_gene, selected_error)


        return selected_gene

    def predict(self, X_test):
        # Controlla se X_test è un DataFrame e converti in tensore
        if isinstance(X_test, pd.DataFrame):
            X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)

        # Modalità valutazione (no gradiente)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test)

        return logits

    def filter(self, data, clusters):
        selection = []
        n_clusters = max(clusters['Cluster']) + 1

        self.logger.log_message(f'Iniziando filtraggio con autoencoder per {n_clusters} cluster')

        for i in range(0, n_clusters): #TODO: parallelizzare il codice
            print(f'Filtering Cluster {i}')
            selected_genes = self.fit(data, clusters, i)
            if selected_genes != None:
                selection.append(selected_genes)

            self.logger.log_message(f'Filtered Clusters {i}/{n_clusters - 1}')
        # Log riassuntivo finale
        self.logger.log_selected_features_summary(selection)
        self.logger.log_completion()
        
        return selection