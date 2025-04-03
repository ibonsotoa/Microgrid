import sys
from pymgrid import MicrogridGenerator as mg
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import zipfile
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

if os.path.exists("data.zip"):
    with zipfile.ZipFile("data.zip", 'r') as zip_ref:
        zip_ref.extractall("./")

env_gen = mg(nb_microgrid=5, path="./data/pymgrid_data")
env_gen.generate_microgrid(verbose=False)
mg0 = env_gen.microgrids[0]

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

env = MicrogridEnv(mg0)
env = Monitor(env, filename="./logs/monitor_qlearning.csv")
check_env(env)

def discretize_state(state, bins_net, bins_soc):
    net_load, soc = state
    net_load_bin = np.digitize(net_load, bins_net) - 1
    soc_bin = np.digitize(soc, bins_soc) - 1
    net_load_bin = np.clip(net_load_bin, 0, len(bins_net) - 2)
    soc_bin = np.clip(soc_bin, 0, len(bins_soc) - 2)
    return (net_load_bin, soc_bin)

low_obs = env.observation_space.low
high_obs = env.observation_space.high
bins_net = np.linspace(low_obs[0], high_obs[0], 20)
bins_soc = np.linspace(low_obs[1], high_obs[1], 11)

n_episodes = 5000
max_steps = env.unwrapped.horizon
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
n_actions = env.action_space.n

Q_table = np.zeros((len(bins_net) - 1, len(bins_soc) - 1, n_actions))
episode_rewards = []

for episode in range(n_episodes):
    state, _ = env.reset()
    state_disc = discretize_state(state, bins_net, bins_soc)
    total_reward = 0

    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q_table[state_disc[0], state_disc[1]])
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        next_state_disc = discretize_state(next_state, bins_net, bins_soc)

        best_next = np.max(Q_table[next_state_disc[0], next_state_disc[1]])
        Q_table[state_disc[0], state_disc[1], action] += alpha * (reward + gamma * best_next - Q_table[state_disc[0], state_disc[1], action])
        state_disc = next_state_disc

        if done:
            break

    episode_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

np.save("q_table.npy", Q_table)

plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Q-Learning Rewards over Episodes")
plt.show()
