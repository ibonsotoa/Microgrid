import streamlit as st
import zipfile
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from pymgrid import MicrogridGenerator as mg
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
import pandas as pd

# -----------------------------
# Definición de la clase MicrogridEnv
# -----------------------------
def actions_agent(mg0, action):
    pv = mg0.pv
    load = mg0.load
    net_load = load - pv

    # Calculamos parámetros de la bateria ###
    # Parámetros para la carga:
    capa_to_charge = mg0.battery.capa_to_charge  # remaining capacity to charge of the battery
    p_charge_max = mg0.battery.p_charge_max  # charge speed of the battery
    p_charge = max(0,min(-net_load, capa_to_charge, p_charge_max))  # Valor de carga para el periodo de tiempo definido (time stamp) charge value for the time
    # Parámetros para la descarga
    capa_to_discharge = mg0.battery.capa_to_discharge  # capacity of discharge
    p_discharge_max = mg0.battery.p_discharge_max  # per hour discharge rate
    p_discharge = max(0,min(net_load, capa_to_discharge, p_discharge_max))  # discharge value for the time

    if action == 0:
        control_dict = {'pv_consummed': min(pv,load),
                        'battery_charge': p_charge,
                        'battery_discharge': 0,
                        'grid_import': 0,
                        'grid_export':max(0,pv - min(pv,load) - p_charge)
                       }
    elif action ==1:
        control_dict = {'pv_consummed': min(pv,load),
                        'battery_charge': 0,
                        'battery_discharge': p_discharge,
                        'grid_import': max(0,load - min(pv,load) - p_discharge),
                        'grid_export':0
                       }
    elif action ==2:
        control_dict = {'pv_consummed': min(pv,load),
                        'battery_charge': 0,
                        'battery_discharge': 0,
                        'grid_import': abs(net_load),
                        'grid_export':0
                       }
    elif action == 3:
        control_dict = {'pv_consummed': min(pv,load),
                        'battery_charge': 0,
                        'battery_discharge': 0,
                        'grid_import': 0,
                        'grid_export':abs(net_load)
                       }

    return control_dict

class MicrogridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, microgrid, horizon=24*7):
        super().__init__()
        self.microgrid = microgrid
        self.horizon = horizon
        self.current_step = 0

        forecast_load = self.microgrid.forecast_load()
        forecast_pv = self.microgrid.forecast_pv()
        net_load = forecast_load - forecast_pv

        self.observation_space = spaces.Box(
            low=np.array([0.9459052, 0.0], dtype=np.float32),
            high=np.array([3.9066246, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        self.microgrid.reset()
        self.current_step = 0
        net_load = self.microgrid.load - self.microgrid.pv
        soc = self.microgrid.battery.soc
        return np.array([net_load, soc], dtype=np.float32), {}

    def step(self, action):
        control_dict = actions_agent(self.microgrid, action)
        self.microgrid.run(control_dict)

        cost = self.microgrid.get_cost()
        reward = -cost

        self.current_step += 1
        done = self.current_step >= self.horizon

        net_load = self.microgrid.load - self.microgrid.pv
        soc = self.microgrid.battery.soc
        observation = np.array([net_load, soc], dtype=np.float32)

        return observation, reward, done, False, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Load: {self.microgrid.load - self.microgrid.pv}, Battery SOC: {self.microgrid.battery.soc}")

    def close(self):
        pass

# -----------------------------
# Funciones de apoyo
# -----------------------------
def extract_zip(file_obj, extract_to="data"):
    """
    Extrae el contenido de un archivo ZIP en una carpeta temporal.
    Si la carpeta ya existe, se elimina antes de extraer.
    """
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.mkdir(extract_to)
    with zipfile.ZipFile(file_obj) as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


def run_test(model, env, num_episodes=1):
    """
    Ejecuta el testeo del modelo en el entorno durante un número de episodios.
    Retorna dos listas: 
      - rewards: la recompensa total obtenida en cada episodio.
      - logs: un registro detallado de cada paso para cada episodio.
    Se usa la API de Monitor, por lo que reset devuelve (obs, info)
    y step devuelve (obs, reward, done, truncation, info).
    """
    rewards = []
    logs = []  # Lista para almacenar el log de cada episodio
    for ep in range(num_episodes):
        episode_log = []  # Log específico del episodio actual
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_counter = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            # Agregamos información del paso al log
            episode_log.append(f"Step {step_counter}: acción = {action}, recompensa = {reward:.2f}")
            step_counter += 1
        rewards.append(total_reward)
        episode_summary = f"Episodio {ep + 1}: Recompensa total = {total_reward:.2f}\n" + "\n".join(episode_log)
        logs.append(episode_summary)
    return rewards, logs
# -----------------------------
# Aplicación Streamlit
# -----------------------------
def main():
    st.title("Optimización de Microgrid con agente PPO")
    st.write("Sube el archivo **data.zip** para extraer los datos de testeo.")

    uploaded_zip = st.file_uploader("Cargar archivo data.zip", type=["zip"])
    
    if uploaded_zip is not None:
        st.info("Extrayendo el archivo...")
        data_folder = extract_zip(uploaded_zip)
        st.success(f"Datos extraídos en la carpeta: `{data_folder}`")
        
        st.info("Inicializando el entorno de Microgrid...")
        try:
            # Inicializamos el entorno pasando la carpeta con los datos extraídos.
            env_gen = mg.MicrogridGenerator(nb_microgrid=5, path=r"C:\Users\Adrian\Documents\MONDRAGON\PBL-3\Microgrid\PMP_Prediction\data\data\pymgrid_data")
            env_gen.generate_microgrid(verbose=False)
            mg0 = env_gen.microgrids[0]
            env = MicrogridEnv(mg0)
            env= Monitor(env)
        except Exception as e:
            st.error("Error al inicializar el entorno de Microgrid. Detalle: " + str(e))
            return
        
        st.info("Cargando el modelo PPO entrenado...")
        try:
            # Se carga el modelo PPO; se asume que el archivo ppo_microgrid_best.zip
            # se encuentra en el mismo directorio que este script.
            model = PPO.load(r"C:\Users\Adrian\Documents\MONDRAGON\PBL-3\Microgrid\ReinforcementLearning\Docker_PPO\ppo_microgrid_best.zip", env=env)
            st.success("Modelo cargado exitosamente.")
        except Exception as e:
            st.error("Error al cargar el modelo PPO. Detalle: " + str(e))
            return

        st.write("Ejecutando test en el entorno...")
        rewards, logs = run_test(model, env, num_episodes=1)
        
        st.subheader("Resultados de la optimización del microgrid")
        for i, r in enumerate(rewards, 1):
            st.success(f"✅Recompensa total = {r:.2f}")
        action_descriptions = {
            0: "Usar PV para carga de batería y exportar excedente a la red.",
            1: "Usar PV y batería para satisfacer la carga, importar si es necesario.",
            2: "Importar energía de la red para satisfacer la carga.",
            3: "Exportar excedente de energía a la red."
        }

        # Convert logs into a DataFrame
        all_logs = []
        for ep_index, log in enumerate(logs, start=1):
            for line in log.split("\n"):
                if "acción =" in line:
                    step = int(line.split("Step")[1].split(":")[0].strip())
                    action = int(line.split("acción =")[1].split(",")[0].strip())
                    reward = float(line.split("recompensa =")[1].strip())
                    description = action_descriptions.get(action, "Acción desconocida")
                    all_logs.append({
                    "Paso": step,
                    "Acción": action,
                    "Descripción": description,
                    "Recompensa": reward
                    })

        df_logs = pd.DataFrame(all_logs)

        # Display the DataFrame in Streamlit
        st.subheader("Logs de las acciones propuestas por el agente")
        st.dataframe(df_logs, width=1200, height=600)

        env.close()

if __name__ == "__main__":
    main()
