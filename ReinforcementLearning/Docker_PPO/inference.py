import gymnasium as gym
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pymgrid import MicrogridGenerator as mg
from gymnasium import spaces

if os.path.exists("data.zip"):
    with zipfile.ZipFile("data.zip", 'r') as zip_ref:
        zip_ref.extractall("./")

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
        control_dict = {
            'pv_consummed': min(pv, load),
            'battery_charge': p_charge,
            'battery_discharge': 0,
            'grid_import': 0,
            'grid_export': max(0, pv - min(pv, load) - p_charge)
        }
    elif action == 1:
        control_dict = {
            'pv_consummed': min(pv, load),
            'battery_charge': 0,
            'battery_discharge': p_discharge,
            'grid_import': max(0, load - min(pv, load) - p_discharge),
            'grid_export': 0
        }
    elif action == 2:
        control_dict = {
            'pv_consummed': min(pv, load),
            'battery_charge': 0,
            'battery_discharge': 0,
            'grid_import': abs(net_load),
            'grid_export': 0
        }
    elif action == 3:
        control_dict = {
            'pv_consummed': min(pv, load),
            'battery_charge': 0,
            'battery_discharge': 0,
            'grid_import': 0,
            'grid_export': abs(net_load)
        }
    else:
        control_dict = {}
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
            low=np.array([net_load.min(), 0.0], dtype=np.float32),
            high=np.array([net_load.max(), 1.0], dtype=np.float32),
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

def main():
    env_gen = mg(nb_microgrid=5, path="./data/pymgrid_data")
    env_gen.generate_microgrid(verbose=False)
    microgrid = env_gen.microgrids[0]

    env = MicrogridEnv(microgrid)
    obs, _ = env.reset()

    if not os.path.exists("ppo_microgrid_best.zip"):
        raise FileNotFoundError("No se encontró el archivo del modelo 'ppo_microgrid_best.zip'")
    model = PPO.load("ppo_microgrid_best")

    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
    print("Recompensa total del episodio:", total_reward)
    env.close()

if __name__ == "__main__":
    main()
