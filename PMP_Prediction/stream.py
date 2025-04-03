import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Función para generar los gráficos de resultados
def plot_results(y_test_np, y_pred, titulo_extra=""):
    """
    Genera tres gráficos:
      1. Scatter plot de Predicciones vs Reales.
      2. Histograma de los residuos.
      3. Serie temporal de Predicciones vs Reales.
    Se puede agregar un título extra en cada figura.
    """
    plt.figure(figsize=(12, 4))
    
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
    return plt

# Función para simular datos (esto debería reemplazarse por la inferencia real de cada modelo)
def simulate_data():
    # Datos de ejemplo: valores reales y predicciones con algo de ruido
    y_test_np = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 1, 100)
    y_pred = y_test_np + noise
    return y_test_np, y_pred

def main():
    st.sidebar.title("Selecciona la Vista")
    vista = st.sidebar.radio("Elige la vista:", ["Server", "Cocoa", "Eugene", "Golden"])

    if vista == "Server":
        st.title("Modelo Federado en el Server")
        st.image(".\images\server.png")  # Reemplaza con la imagen que desees
        st.write("Resultados de inferencia:")
        y_test, y_pred = simulate_data()
        fig = plot_results(y_test, y_pred, titulo_extra=" - Server")
        st.pyplot(fig.gcf())
        
    elif vista == "Cocoa":
        st.title("Ubicación:")
        st.image(".\images\Cocoa.png")  # Reemplaza con la imagen correspondiente
        st.write("Resultados de inferencia:")
        y_test, y_pred = simulate_data()
        fig = plot_results(y_test, y_pred, titulo_extra=" - Cocoa")
        st.pyplot(fig.gcf())
        
    elif vista == "Eugene":
        st.title("Ubicación: ")
        st.image(".\images\Eugene.png")  # Reemplaza con la imagen correspondiente
        st.write("Resultados de inferencia:")
        y_test, y_pred = simulate_data()
        fig = plot_results(y_test, y_pred, titulo_extra=" - Eugene")
        st.pyplot(fig.gcf())
        
    elif vista == "Golden":
        st.title("Ubicación:")
        st.image(".\images\Golden.png")  # Reemplaza con la imagen correspondiente
        st.write("Resultados de inferencia:")
        y_test, y_pred = simulate_data()
        fig = plot_results(y_test, y_pred, titulo_extra=" - Golden")
        st.pyplot(fig.gcf())

if __name__ == '__main__':
    main()
