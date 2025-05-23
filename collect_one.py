import cv2
import os
import os.path as osp
import multiprocessing as mp
import gym
import numpy as np
import argparse
from env import CoordinateExplore

def get_action(corner_nums, corner_wait, forward_step, jump_prob):
    assert corner_nums in [2, 3, 6]
    left = np.random.choice([True, False], p=[0.5, 0.5])
    all_returns = []
    for corner in range(corner_nums):
        for i in range(forward_step):
            jump = np.random.choice([0, 1], p=[1 - jump_prob, jump_prob])
            action = dict(forward=np.array(1), back=np.array(0), jump=np.array(jump), camera=np.array([0., 0.]))
            action_list = [1, 0, jump, 0]
            all_returns.append((action, action_list))
        turn_degree = 360 / corner_nums
        turn_step_num = int(turn_degree / 20)
        for i in range(turn_step_num):
            if left:
                action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., -20.]))
                action_list = [0, 0, 0, -1]
            else:
                action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., 20.]))
                action_list = [0, 0, 0, 1]
            all_returns.append((action, action_list))
        for i in range(corner_wait):
            action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., 0.]))
            action_list = [0, 0, 0, 0]
            all_returns.append((action, action_list))
    return all_returns

def get_fix_action():
    left = np.random.choice([True, False], p=[0.5, 0.5])
    all_returns = []
    turn_step_num = int(100 / 20)
    for i in range(turn_step_num):
        if left:
            action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., -20.]))
            action_list = [0, 0, 0, -1]
        else:
            action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., 20.]))
            action_list = [0, 0, 0, 1]
        all_returns.append((action, action_list))

    forward_step = 10
    for i in range(forward_step):
        action = dict(forward=np.array(1), back=np.array(0), jump=np.array(0), camera=np.array([0., 0.]))
        action_list = [1, 0, 0, 0]
        all_returns.append((action, action_list))

    turn_step_num = int(180 / 20)
    for i in range(turn_step_num):
        if left:
            action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., -20.]))
            action_list = [0, 0, 0, -1]
        else:
            action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., 20.]))
            action_list = [0, 0, 0, 1]
        all_returns.append((action, action_list))

    forward_step = 10
    for i in range(forward_step):
        action = dict(forward=np.array(1), back=np.array(0), jump=np.array(0), camera=np.array([0., 0.]))
        action_list = [1, 0, 0, 0]
        all_returns.append((action, action_list))

    turn_step_num = int(100 / 20)
    for i in range(turn_step_num):
        if not left:
            action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., -20.]))
            action_list = [0, 0, 0, -1]
        else:
            action = dict(forward=np.array(0), back=np.array(0), jump=np.array(0), camera=np.array([0., 20.]))
            action_list = [0, 0, 0, 1]
        all_returns.append((action, action_list))

    return all_returns

def collect_episode(env, required_actions):
    traj_length = len(required_actions)
    obs = env.reset()
    observations = [obs['pov']] 
    actions = [np.array([0, 0, 0, 0, 0, 0, 0, 0])]
    for t in range(traj_length):
        action, a_id = required_actions[t]
        obs, _, done, _ = env.step(action)
        observations.append(obs['pov'])
        actions.append(np.array(a_id + [obs['location_stats']['xpos'].astype(float), obs['location_stats']['ypos'].astype(float), obs['location_stats']['zpos'].astype(float), obs['location_stats']['yaw'].astype(float)]))
        if done and t < traj_length - 1: # Invalid if the agent dies early
            return None

    rgb = np.stack(observations, axis=0)
    actions = np.stack(actions, axis=0)
    actions = actions.astype(np.float32)
    return rgb, actions

def main(args):
    # logs_dir = "logs"
    # if os.path.exists(logs_dir):
    #     os.system(f'rm -r {logs_dir}')
    abs_env = CoordinateExplore(resolution=(args.reso_w, args.reso_h),
                            biomes=[6])
    abs_env.register()
    os.makedirs(args.output_dir, exist_ok=True)
    env = gym.make('CoordinateExplore-v0')
    # agent = SimpleAgent()

    one_actions = get_action(corner_nums=2, corner_wait=0, forward_step=0, jump_prob=0)
    # one_actions = get_fix_action()

    all_actions = []
    while len(all_actions) < args.traj_length:
        all_actions += one_actions
    all_actions = all_actions[:args.traj_length]
    print(f"Base action length: {len(one_actions)}")

    i = 0
    while i < args.num_episodes:
        video_fname = osp.join(args.output_dir, f'{i:06d}.mp4')
        action_fname = osp.join(args.output_dir, f'{i:06d}.npz')
        if osp.exists(video_fname) or osp.exists(action_fname):
            if not (osp.exists(video_fname) and osp.exists(action_fname)):
                print(f"Warning: Incomplete files found for episode {i}.")
            i += 1
            continue
        out = collect_episode(env, all_actions)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-t', '--traj_length', type=int, default=150,
                        help='default: 100')
    parser.add_argument('-n', '--num_episodes', type=int, default=100,
                        help='default: 100')
    parser.add_argument('-reso_h', type=int, default=360)
    parser.add_argument('-reso_w', type=int, default=640)
    args = parser.parse_args()

    main(args)
