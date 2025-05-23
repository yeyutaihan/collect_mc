import cv2
import os
import os.path as osp
import multiprocessing as mp
import gym
import numpy as np
import argparse
from env import CoordinateExplore


class SimpleAgent:
    def __init__(self):
        self.action_repeat = 5
        self.prob_forward = 0.95
        self.prob_backward = 0.0
        self.prob_turn = 0.25
        self.prob_jump = 0.2
        self.reset()


    def reset(self):
        self.counter = 0
        self.action = None

    def sample_action(self):
        forward_backward = np.random.choice([-1, 0, 1], p=[self.prob_backward, 1 - self.prob_forward - self.prob_backward, self.prob_forward])
        forward = 1 if forward_backward == 1 else 0
        back = 1 if forward_backward == -1 else 0
        turn = np.random.choice([-1, 0, 1], p=[self.prob_turn, 1 - 2 * self.prob_turn, self.prob_turn])
        jump = np.random.choice([0, 1], p=[1 - self.prob_jump, self.prob_jump])
        action = dict(forward=np.array(forward), back=np.array(back), jump=np.array(jump), camera=np.array([0., turn * 20.]))
        action_list = [forward, back, jump, turn]
        return action, action_list
    def sample(self):
        if self.action is None or self.counter % self.action_repeat == 0:
            self.action = self.sample_action()

        self.counter += 1
        return self.action


ACTIONS = {
    'forward': dict(forward=np.array(1), jump=np.array(1), camera=np.array([0., 0.])),
    'left': dict(forward=np.array(0), jump=np.array(1), camera=np.array([0., -20.])),
    'right': dict(forward=np.array(0), jump=np.array(1), camera=np.array([0., 20.])),
    'noop': dict(forward=np.array(0), jump=np.array(0), camera=np.array([0., 0.]))
}

ACTIONS_TO_ID = {
    'forward': 0,
    'left': 1,
    'right': 2,
}


def sample_action_old(prob_forward):
    prob_turn = (1 - prob_forward) / 2
    i = np.random.choice(['forward', 'left', 'right'],
                         p=[prob_forward, prob_turn, prob_turn])
    return ACTIONS[i], ACTIONS_TO_ID[i]

def collect_episode(env, agent, traj_length):
    agent.reset()
    obs = env.reset()
    observations = [obs['pov']] 
    actions = [np.array([0, 0, 0, 0, 0, 0, 0, 0])]
    for t in range(traj_length):
        action, a_id = agent.sample()
        obs, _, done, _ = env.step(action)
        observations.append(obs['pov'])
        actions.append(np.array(a_id + [obs['location_stats']['xpos'].astype(float), obs['location_stats']['ypos'].astype(float), obs['location_stats']['zpos'].astype(float), obs['location_stats']['yaw'].astype(float)]))
        if done and t < traj_length - 1: # Invalid if the agent dies early
            return None

    rgb = np.stack(observations, axis=0)
    actions = np.stack(actions, axis=0)
    actions = actions.astype(np.float32)
    return rgb, actions


def worker(id, args):
    args.output_dir = osp.join(args.output_dir, f'{id}')
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make('CoordinateExplore-v0')
    agent = SimpleAgent()

    num_episodes = args.num_episodes // args.n_parallel + (id < (args.num_episodes % args.n_parallel))
    # pbar = tqdm(total=num_episodes, position=id)
    i = 0
    while i < num_episodes:
        video_fname = osp.join(args.output_dir, f'{i:06d}.mp4')
        action_fname = osp.join(args.output_dir, f'{i:06d}.npz')
        if osp.exists(video_fname) or osp.exists(action_fname):
            if not (osp.exists(video_fname) and osp.exists(action_fname)):
                print(f"Warning: Incomplete files found for episode {i}.")
            i += 1
            continue
        out = collect_episode(env, agent, args.traj_length)
        if out is None:
            continue

        rgb, actions = out
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_fname, fourcc, 20.0, (rgb.shape[2], rgb.shape[1]), True)
        for t in range(rgb.shape[0]):
            frame = rgb[t]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

        np.savez_compressed(action_fname, actions=actions)
        i += 1


def main(args):
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        os.system(f'rm -r {logs_dir}')
    abs_env = CoordinateExplore(resolution=(args.reso_w, args.reso_h),
                            biomes=[6])
    abs_env.register()

    os.makedirs(args.output_dir, exist_ok=True)

    procs = [mp.Process(target=worker, args=(i, args)) for i in range(args.n_parallel)]
    [p.start() for p in procs]
    [p.join() for p in procs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-z', '--n_parallel', type=int, default=24,
                        help='default: 1')
    parser.add_argument('-t', '--traj_length', type=int, default=1200,
                        help='default: 100')
    parser.add_argument('-n', '--num_episodes', type=int, default=6000,
                        help='default: 100')
    parser.add_argument('-reso_h', type=int, default=360)
    parser.add_argument('-reso_w', type=int, default=640)
    args = parser.parse_args()

    main(args)
