<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <!-- Carga de fuente: puedes cambiar a la fuente que prefieras -->
  <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
</head>
<body>

# Sistema de Gestión de Microred Eléctrica basado en Reinforcement Learning y Machine Learning

Este repositorio contiene el código y recursos para la implementación de un sistema inteligente orientado a la optimización de costes y la predicción de la eficiencia de una microred eléctrica. Se emplean algoritmos de Reinforcement Learning (RL) y técnicas de Deep Learning y Aprendizaje Federado para abordar distintos objetivos del proyecto.

## Introducción

El proyecto se centra en dos grandes objetivos:

- **Optimización de costes**: Utilización de algoritmos de RL (A2C, PPO y Q-learning) para maximizar la ganancia monetaria de la microred, entrenando agentes en el entorno PyMGrid y ajustando hiperparámetros (utilizando herramientas como Optuna) para mejorar el rendimiento.
- **Predicción de potencia máxima**: Desarrollo de un modelo que predice la potencia máxima (Pmp) de módulos fotovoltaicos. Se compara un enfoque centralizado con un sistema de aprendizaje federado, preservando la privacidad de los datos de tres clientes y realizando fine tuning para mejorar la precisión en el cliente subrepresentado.

El proyecto también contempla la validación de requisitos y especificaciones definidos en colaboración con 5 stakeholders, y la dockerización de la solución para facilitar su despliegue en la nube mediante una interfaz basada en Streamlit.

## Objetivos

- **Optimización de Costes en la Microred**:
  - Implementar y evaluar distintos algoritmos de RL (A2C, PPO, Q-learning).
  - Ajustar hiperparámetros para obtener los mejores rewards (ejemplo: -146,77 para A2C y PPO; -448,65 para Q-learning).
  
- **Predicción de la Potencia Máxima de Módulos Fotovoltaicos**:
  - Entrenar un modelo global con dataset centralizado (R2: 0.99) y evaluar la pérdida de precisión al privatizar los datos con aprendizaje federado (R2: 0.83).
  - Mejorar la precisión mediante fine tuning de submodelos, ajustando la performance en cada cliente (antes de FT: 0.89, 0.88, 0.35; después de FT: 0.87, 0.85, 0.86).

- **Integración y Despliegue**:
  - Dockerización de todos los componentes (modelos de RL y ML federado) para su futura implementación en la nube.
  - Desarrollo de tests metamórficos para garantizar la calidad y robustez de los algoritmos.

## Estructura del Repositorio

- **Exploration**  
  Contiene el análisis exploratorio de datos (EDA) aplicado al dataset relativo al análisis de placas fotovoltaicas en la microred.

- **Model testing / DeepLearning**  
  Incluye los modelos de deep learning utilizados para testear y validar el código.

- **objective1_data**  
  Datos para el Objetivo 1 (Reinforcement Learning). Se encuentra organizado en:
  - `pymgrid_data/co2`
  - `pymgrid_data/load`
  - `pymgrid_data/pv`

- **objetive2_data**  
  Datos correspondientes al Objetivo 2, orientados al aprendizaje federado para 3 clientes.

- **objetive2_data_cleaned**  
  Datos procesados en los notebooks de la carpeta `PMP_Prediction`. Dentro se incluyen:
  - **images**: Imágenes relacionadas con la parte de aprendizaje federado.
  - **models**: Modelos generados durante las diferentes fases de desarrollo.
  - **notebooks**: Notebooks que contienen el desarrollo de modelos de deep learning, tanto centralizados como federados.

- **ReinforcementLearning**  
  Relacionada con el Objetivo 1, esta carpeta se compone de:
  - **Basecode**: Notebook utilizado para el entrenamiento de los agentes de RL.
  - **Docker_A2C**, **Docker_PPO**, **Docker_QLearning**: Contienen los archivos Docker para cada uno de los algoritmos de RL.
  - **Testing**: Notebooks con tests metamórficos diseñados para evaluar los diferentes algoritmos de RL.

- **TrainTestSplit**  
  Notebook empleado para realizar los diferentes train-test splits utilizados en el Objetivo 2.

## Metodología

- **Reinforcement Learning**:  
  Se entrenaron agentes utilizando tres algoritmos (A2C, PPO y Q-learning) en el entorno PyMGrid, con ajustes de hiperparámetros mediante Optuna. Los resultados de los rewards finales indican una mejor consistencia en A2C y PPO frente a Q-learning.

- **Aprendizaje Federado para Predicción de Pmp**:  
  Se comparó la precisión entre un modelo global centralizado (R2: 0.99) y un sistema federado que, inicialmente, mostró una precisión reducida (R2: 0.83). El fine tuning de los submodelos mejoró significativamente la performance en clientes subrepresentados.

- **Validación de Requisitos y Especificaciones**:  
  El sistema fue desarrollado considerando 7 requisitos funcionales y 6 no funcionales, garantizando que la solución cumple con las expectativas de todos los stakeholders involucrados.

- **Dockerización y Despliegue**:  
  Todos los componentes del proyecto se dockerizaron para facilitar su despliegue en entornos de nube. Además, se implementaron tests locales mediante un pipeline simulado para validar la integridad del despliegue.

## Resultados

- **Optimización de Costes**:  
  Los rewards finales obtenidos fueron:
  - A2C: -146,77  
  - PPO: -146,77  
  - Q-learning: -448,65

- **Predicción de Potencia Máxima**:  
  - Modelo global (centralizado): R2 de 0.99  
  - Sistema federado: R2 de 0.83, con mejoras posteriores mediante fine tuning de submodelos.

## Requisitos y Especificaciones del Sistema

- Se tuvieron en cuenta los requerimientos de 5 stakeholders.
- Se definieron 7 requisitos funcionales y 6 no funcionales que rigen la implementación y el desempeño del sistema.

## Despliegue y Dockerización

- **Dockerización**:  
  Todos los modelos de RL y la solución de aprendizaje federado están dockerizados, permitiendo su despliegue en la nube y facilitando el escalado del sistema.

- **Interfaz de Despliegue**:  
  Se ha desarrollado una aplicación basada en Streamlit para el despliegue y la visualización de los resultados del sistema.

## Autores

- **Beñat Alkain**
- **Adrián Ruiz**
- **Ibon Soto**

Máster en Inteligencia Artificial Aplicada
