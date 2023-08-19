# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Examples for vision environments."""
from PIL import Image
import argparse
import os
import re

import safety_gymnasium
from gymnasium.utils.save_video import save_video


DIR = os.path.join(os.path.dirname(__file__), 'demoimg')


def run_random(env_name):
    """Random run."""
    env = safety_gymnasium.make(env_name, render_mode='rgb_array', width=2560, height=2560, camera_name="fixedfar")
    obs, _ = env.reset()
    # Use below to specify seed.
    # obs, _ = env.reset(seed=0)
    terminated, truncated = {'agent_0': False}, {'agent_0': False}
    ep_ret, ep_cost = 0, 0
    for i in range(1):  # pylint: disable=unused-variable
        if terminated['agent_0'] or truncated['agent_0']:
            print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()

        act = {}
        for agent in env.agents:
            assert env.observation_space(agent).contains(obs[agent])
            act[agent] = env.action_space(agent).sample()
            assert env.action_space(agent).contains(act[agent])

        obs, reward, cost, terminated, truncated, _ = env.step(act)

        for agent in env.agents:
            ep_ret += reward[agent]
            ep_cost += cost[agent]
        img = env.render()
        img = Image.fromarray(img)
        os.makedirs(DIR, exist_ok=True)
        # img.save(os.path.join(DIR, f"{re.findall('[A-Z][^A-Z]*', env_name)[2].lower().split('-')[0]}.jpeg"))
        img.save(os.path.join(DIR, f"{env_name}.jpeg"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyRacecarGoal2Vision-v0')
    args = parser.parse_args()
    agent_id=['Point', 'Ant']
    env_id=['MultiGoal']
    level=['0']
    for agent in agent_id:
        for env in env_id:
            for l in level:
                # print(agent, env, l)
                env_name = 'Safety' + agent + env + l + '-v0'
                run_random(env_name)
