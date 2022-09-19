import argparse
import os
import sys
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
import traci


if __name__ == '__main__':

    alpha = 0.1
    gamma = 0.995
    decay = 1
    runs = 4

    env = SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                          route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                          use_gui=True,
                          num_seconds=80000,
                          min_green=5,
                          delta_time=5)

    initial_states = env.reset()
    ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)) for ts in env.ts_ids}
    counter = 0
    for run in range(1, runs+1):
        if run != 1:
            initial_states = env.reset()
            for ts in initial_states.keys():
                ql_agents[ts].state = env.encode(initial_states[ts], ts)

        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            if '4.100' in traci.vehicle.getIDList() and counter < 100:
                traci.vehicle.setStop(vehID='4.100', edgeID='0to1', pos=70, duration=10)
                counter += 1
                #t = traci.vehicle.getWaitingTime(vehID='0.100')
                #print('waiting time', t)
                print('Pos', traci.vehicle.getPosition(vehID='4.100'))
            else:
                pass

            s, r, done, info = env.step(action=actions)
            
            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        env.save_csv('outputs/4x4/ql-test-995speed', run)
        env.close()


