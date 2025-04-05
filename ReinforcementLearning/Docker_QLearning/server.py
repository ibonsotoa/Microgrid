from flask import Flask, request, jsonify
import numpy as np
import zipfile
import os
from pymgrid import MicrogridGenerator as mg
from gymnasium import spaces
import gymnasium as gym

# Descomprimir datos si no están descomprimidos
if os.path.exists("data.zip") and not os.path.exists("data/pymgrid_data"):
    with zipfile.ZipFile("data.zip", 'r') as zip_ref:
        zip_ref.extractall("./")

# Inicializar microgrid
env_gen = mg(nb_microgrid=5, path="./data/pymgrid_data")
env_gen.generate_microgrid(verbose=False)
mg0 = env_gen.microgrids[0]

# Definición de entorno personalizado
class MicrogridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, microgrid, horizon=24*7):
        super(MicrogridEnv, self).__init__()
        self.microgrid = microgrid
        self.horizon = horizon
        self.current_step = 0

        forecast_load = self.microgrid.forecast_load()
        forecast_pv = self.microgrid.forecast_pv()
        net_load = forecast_load - forecast_pv

        self.observation_space = spaces.Box(
            low=np.array([net_load.min(), 0.0], dtype=np.float32),
            high=np.array([net_load.max(), 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def reset(self):
        self.microgrid.reset()
        self.current_step = 0
        net_load = self.microgrid.load - self.microgrid.pv
        soc = self.microgrid.battery.soc
        return np.array([net_load, soc], dtype=np.float32)

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

        return observation, reward, done, {}

def actions_agent(mg0, action):
    pv = mg0.pv
    load = mg0.load
    net_load = load - pv

    capa_to_charge = mg0.battery.capa_to_charge
    p_charge_max = mg0.battery.p_charge_max
    p_charge = max(0, min(-net_load, capa_to_charge, p_charge_max))

    capa_to_discharge = mg0.battery.capa_to_discharge
    p_discharge_max = mg0.battery.p_discharge_max
    p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))

    if action == 0:
        control_dict = {'pv_consummed': min(pv, load),
                        'battery_charge': p_charge,
                        'battery_discharge': 0,
                        'grid_import': 0,
                        'grid_export': max(0, pv - min(pv, load) - p_charge)}
    elif action == 1:
        control_dict = {'pv_consummed': min(pv, load),
                        'battery_charge': 0,
                        'battery_discharge': p_discharge,
                        'grid_import': max(0, load - min(pv, load) - p_discharge),
                        'grid_export': 0}
    elif action == 2:
        control_dict = {'pv_consummed': min(pv, load),
                        'battery_charge': 0,
                        'battery_discharge': 0,
                        'grid_import': abs(net_load),
                        'grid_export': 0}
    elif action == 3:
        control_dict = {'pv_consummed': min(pv, load),
                        'battery_charge': 0,
                        'battery_discharge': 0,
                        'grid_import': 0,
                        'grid_export': abs(net_load)}
    else:
        control_dict = {}
    return control_dict

# Discretización
def discretize_state(state, bins_net, bins_soc):
    net_load, soc = state
    net_load_bin = np.digitize(net_load, bins_net) - 1
    soc_bin = np.digitize(soc, bins_soc) - 1
    net_load_bin = np.clip(net_load_bin, 0, len(bins_net) - 2)
    soc_bin = np.clip(soc_bin, 0, len(bins_soc) - 2)
    return (net_load_bin, soc_bin)

# Inicializar entorno
env = MicrogridEnv(mg0)

low_obs = env.observation_space.low
high_obs = env.observation_space.high
bins_net = np.linspace(low_obs[0], high_obs[0], 20)
bins_soc = np.linspace(low_obs[1], high_obs[1], 11)

# Cargar tabla Q
Q_table = np.load("q_table.npy")

# Crear app Flask
app = Flask(__name__)

@app.route("/inference", methods=["POST"])
def infer():
    try:
        state = env.reset()
        state_disc = discretize_state(state, bins_net, bins_soc)
        action = int(np.argmax(Q_table[state_disc[0], state_disc[1]]))
        return jsonify({"action": action})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
