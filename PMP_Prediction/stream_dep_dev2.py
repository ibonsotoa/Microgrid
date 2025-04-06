import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import copy
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt

# Configuración inicial de Streamlit
st.set_page_config(page_title="Entrenamiento Federado", layout="wide")
st.title("Sistema Federado para Microgrids")

# Clase del modelo
class EnhancedDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)
def plot_results(y_test_np, y_pred, titulo_extra=""):
    fig = plt.figure(figsize=(12, 4))
    
    # Gráfico 1: Predicciones vs Reales
    plt.subplot(131)
    plt.title('Predicciones vs Reales' + titulo_extra)
    plt.scatter(y_test_np, y_pred, alpha=0.3)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.grid(True)
    
    # Gráfico 2: Distribución de Residuos
    plt.subplot(132)
    residuals = y_test_np - y_pred
    plt.hist(residuals, bins=50, density=True, alpha=0.6)
    plt.title('Distribución de Residuos' + titulo_extra)
    plt.xlabel('Residuo')
    plt.grid(True)
    
    # Gráfico 3: Serie temporal de Predicciones vs Reales
    plt.subplot(133)
    plt.plot(y_pred, label="Predicciones")
    plt.plot(y_test_np, label="Valores Reales", alpha=0.7)
    plt.title('Serie Temporal' + titulo_extra)
    plt.xlabel('Muestras')
    plt.legend() 
    plt.grid(True)
    
    plt.tight_layout()
    return fig
# Funciones de ayuda modificadas para los nuevos nombres de archivo
def load_and_preprocess(files, folder_path, all_sources):
    df_list = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df = df.drop(columns='Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss')
        df_sampled = df.groupby('source', group_keys=False).apply(lambda x: x.sample(frac=0.4, random_state=42))
        df_list.append(df_sampled)
    
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['source'] = pd.Categorical(full_df['source'], categories=all_sources)
    full_df = pd.get_dummies(full_df, columns=['source'], prefix='src', dtype=np.float32)
    
    return full_df

def prepare_data(df, scaler=None):
    X = df.drop(columns=['Pmp (W)']).values.astype(np.float32)
    y = df['Pmp (W)'].values.astype(np.float32)
    
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return torch.tensor(X), torch.tensor(y).unsqueeze(1), scaler

def local_train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        st.write(f"Época {epoch+1}, Pérdida: {avg_loss:.4f}")
    return model.state_dict(), len(train_loader.dataset), losses

def fed_avg(state_dicts, data_sizes):
    avg_state = copy.deepcopy(state_dicts[0])
    total_samples = sum(data_sizes)
    
    for key in avg_state.keys():
        if avg_state[key].dtype in [torch.int64, torch.long]:
            avg_state[key] = state_dicts[0][key].clone()
        else:
            avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
            for state, size in zip(state_dicts, data_sizes):
                weight = size / total_samples
                avg_state[key] += state[key].float() * weight
    return avg_state


# Configuración de la barra lateral
st.sidebar.header("Configuración de Hiperparámetros")
federated_rounds = st.sidebar.slider("Rondas federadas", 1, 10, 3)
local_epochs = st.sidebar.slider("Épocas locales", 1, 100, 15)
batch_size = st.sidebar.slider("Tamaño de lote", 32, 1024, 256)
lr = st.sidebar.selectbox(
    "Tasa de aprendizaje",
    options=[1e-5, 1e-4, 1e-3, 1e-2],
    index=2
)

# Configuración de rutas y datos
folder_path = "../TrainTestSplit/"
clients = {
    "cocoa": [f for f in os.listdir(folder_path) if f.startswith('train_cocoa')],
    "eugene": [f for f in os.listdir(folder_path) if f.startswith('train_eugene')],
    "golden": [f for f in os.listdir(folder_path) if f.startswith('train_golden')]
}

test_files = [
    f for f in os.listdir(folder_path) 
    if f.startswith('test_') and any(client in f for client in ["cocoa", "eugene", "golden"])
]

# Inicialización del modelo
if 'global_model' not in st.session_state:
    all_sources = sorted(set().union(*[
        pd.read_csv(os.path.join(folder_path, file), usecols=['source'])['source'].unique()
        for client_files in clients.values() for file in client_files
    ]))
    
    temp_df = load_and_preprocess(clients["cocoa"], folder_path, all_sources)
    X_temp, _, _ = prepare_data(temp_df)
    input_dim = X_temp.shape[1]
    
    st.session_state.global_model = EnhancedDNN(input_dim)
    st.session_state.scaler = None
    st.session_state.client_models = {
        "cocoa": copy.deepcopy(st.session_state.global_model),
        "eugene": copy.deepcopy(st.session_state.global_model),
        "golden": copy.deepcopy(st.session_state.global_model)
    }
    st.session_state.loss_history = {client: [] for client in clients}
    st.session_state.test_losses = []
    st.session_state.maes = []
    st.session_state.r2s = []
    st.session_state.all_sources = all_sources

# Sección de entrenamiento
if st.sidebar.button("Iniciar Entrenamiento"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for round in range(federated_rounds):
        status_text.text(f"Ronda Federada {round+1}/{federated_rounds}")
        progress_bar.progress((round+1)/federated_rounds)
        
        local_state_dicts = []
        data_sizes = []
        
        for client_name in clients.keys():
            with st.expander(f"Cliente: {client_name}", expanded=True):
                st.write(f"### Entrenando cliente: {client_name}")
                
                # Cargar datos
                client_files = clients[client_name]
                client_df = load_and_preprocess(client_files, folder_path, st.session_state.all_sources)
                X_local, y_local, _ = prepare_data(client_df, st.session_state.scaler)
                local_dataset = TensorDataset(X_local, y_local)
                local_loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)
                
                # Entrenamiento local
                local_model = copy.deepcopy(st.session_state.global_model)
                optimizer = optim.Adam(local_model.parameters(), lr=lr)
                criterion = nn.MSELoss()
                
                state_dict, n_samples, losses = local_train(
                    local_model, local_loader, criterion, optimizer, local_epochs
                )
                
                # Actualizar modelo del cliente en session_state
                st.session_state.client_models[client_name].load_state_dict(state_dict)
                local_state_dicts.append(state_dict)
                data_sizes.append(n_samples)
                st.session_state.loss_history[client_name].extend(losses)
        
        # Actualización global
        global_state = fed_avg(local_state_dicts, data_sizes)
        st.session_state.global_model.load_state_dict(global_state)
        
    #     # Evaluación con todos los archivos de test
    #     test_dfs = []
    #     for test_file in test_files:
    #         test_df = load_and_preprocess([test_file], folder_path, st.session_state.all_sources)
    #         test_dfs.append(test_df)
        
    #     full_test_df = pd.concat(test_dfs, ignore_index=True)
    #     X_test, y_test, _ = prepare_data(full_test_df, st.session_state.scaler)
        
    #     with torch.no_grad():
    #         y_pred = st.session_state.global_model(X_test)
    #         test_loss = mean_squared_error(y_test.numpy(), y_pred.numpy())
    #         mae=mean_absolute_error(y_test.numpy(), y_pred.numpy())
    #         r2=r2_score(y_test.numpy(), y_pred.numpy())
    #         st.session_state.test_losses.append(test_loss)
    #         st.session_state.test_maes.append(mae)
    #         st.session_state.test_r2s.append(r2)
    #         st.write(f"**MSE después de ronda {round+1}:** {test_loss:.4f}")
    #         st.write(f"**MAE después de ronda {round+1}:** {mae:.4f}")
    #     # Sección nueva de evaluación después del entrenamiento
    # st.success("Entrenamiento completado exitosamente!")
    # st.header("Resultados de Evaluación en Test")
    
        # # Preparar datos de test
        # test_dfs = []
        # for test_file in test_files:
        #     test_df = load_and_preprocess([test_file], folder_path, st.session_state.all_sources)
        #     test_dfs.append(test_df)
        # full_test_df = pd.concat(test_dfs, ignore_index=True)
        # X_test, y_test, _ = prepare_data(full_test_df, st.session_state.scaler)
        # y_test_np = y_test.numpy()
        
        test_dfs_global = []
        for test_file in test_files:
            test_df = load_and_preprocess([test_file], folder_path, st.session_state.all_sources)
            test_dfs_global.append(test_df)
        full_test_df_global = pd.concat(test_dfs_global, ignore_index=True)
        X_test_global, y_test_global, _ = prepare_data(full_test_df_global, st.session_state.scaler)
        y_test_global_np = y_test_global.numpy()
        
        client_test_sets = {}
        for client_name in clients.keys():
            client_test_files = [f for f in test_files if f.startswith(f'test_{client_name}')]
            test_dfs = []
            for test_file in client_test_files:
                test_df = load_and_preprocess([test_file], folder_path, st.session_state.all_sources)
                test_dfs.append(test_df)
            if test_dfs:
                full_test_df = pd.concat(test_dfs, ignore_index=True)
                X_test, y_test, _ = prepare_data(full_test_df, st.session_state.scaler)
                client_test_sets[client_name] = (X_test, y_test.numpy())
        # Crear pestañas para los resultados
        tab1, tab2, tab3, tab4 = st.tabs([
            "Modelo Global", 
            "Cliente Cocoa", 
            "Cliente Eugene", 
            "Cliente Golden"
        ])
        
        with tab1:
            with torch.no_grad():
                y_pred = st.session_state.global_model(X_test_global).numpy()
                fig = plot_results(y_test_global_np, y_pred, " - Modelo Global")
                st.pyplot(fig)
                test_loss = mean_squared_error(y_test_global_np, y_pred)
                mae=mean_absolute_error(y_test_global_np, y_pred)
                r2=r2_score(y_test_global_np, y_pred)
                st.session_state.test_losses.append(test_loss)
                st.session_state.maes.append(mae)
                st.session_state.r2s.append(r2)
                st.write(f"**MSE después de ronda {round+1}:** {test_loss:.4f}")
                st.write(f"**MAE después de ronda {round+1}:** {mae:.4f}")
                st.write(f"**R2 después de ronda {round+1}:** {r2:.4f}")
        for client_name, tab in zip(["cocoa", "eugene", "golden"], [tab2, tab3, tab4]):
            with tab:
                if client_name in client_test_sets:
                    X_test, y_test_np = client_test_sets[client_name]
                    with torch.no_grad():
                        model = st.session_state.client_models[client_name]
                        y_pred = st.session_state.global_model(X_test).numpy()
                        fig = plot_results(y_test_np, y_pred, f" - Cliente {client_name.capitalize()}")
                        test_loss = mean_squared_error(y_test_np, y_pred)
                        mae=mean_absolute_error(y_test_np, y_pred)
                        r2=r2_score(y_test_np, y_pred)
                        st.write(f"**MSE después de ronda {round+1}:** {test_loss:.4f}")
                        st.write(f"**MAE después de ronda {round+1}:** {mae:.4f}")
                        st.write(f"**R2 después de ronda {round+1}:** {r2:.4f}")
                        st.pyplot(fig)
        # with tab2:
        #     with torch.no_grad():
        #         model = st.session_state.client_models["cocoa"]
        #         y_pred = model(X_test).numpy()
        #         fig = plot_results(y_test_np, y_pred, " - Cliente Cocoa")
        #         st.pyplot(fig)
                
        # with tab3:
        #     with torch.no_grad():
        #         model = st.session_state.client_models["eugene"]
        #         y_pred = model(X_test).numpy()
        #         fig = plot_results(y_test_np, y_pred, " - Cliente Eugene")
        #         st.pyplot(fig)
                
        # with tab4:
        #     with torch.no_grad():
        #         model = st.session_state.client_models["golden"]
        #         y_pred = model(X_test).numpy()
        #         fig = plot_results(y_test_np, y_pred, " - Cliente Golden")
        #         st.pyplot(fig)
    st.success("Entrenamiento completado exitosamente!")

# Visualización de resultados
st.header("Resultados del Entrenamiento")
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Progreso del Entrenamiento")
    
    tab1, tab2 = st.tabs(["Pérdidas Locales", "Pérdida Global"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        for client, losses in st.session_state.loss_history.items():
            ax.plot(losses, label=client.capitalize())
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida MSE")
        ax.set_title("Pérdidas de Entrenamiento por Cliente")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    with tab2:
        if st.session_state.test_losses:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(st.session_state.test_losses, marker='o', color='purple')
            ax.set_xlabel("Ronda Federada")
            ax.set_ylabel("Pérdida MSE")
            ax.set_title("Evolución de la Pérdida en Test")
            ax.grid(True)
            st.pyplot(fig)

with col2:
    st.write("**Métrica Final:**")
    if st.session_state.test_losses:
        st.metric(
            label="Mejor Pérdida en Test",
            value=f"{min(st.session_state.test_losses):.4f}",
            delta=f"{(st.session_state.test_losses[0] - st.session_state.test_losses[-1]):.4f}" if len(st.session_state.test_losses) > 1 else None
        )

