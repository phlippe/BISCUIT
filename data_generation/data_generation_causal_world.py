""" 
Dataset generation of the CausalWorld environment
"""

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pyb
from imageio.v3 import imwrite, imread
from tqdm.auto import tqdm
from glob import glob
import os
from collections import defaultdict
from multiprocessing import Pool
from argparse import ArgumentParser
import time
import json
import pybullet
import torch
import torch.nn.functional as F
import sys
import math

from causal_world.envs import CausalWorld
from causal_world.task_generators.task import generate_task
from causal_world.configs.world_constants import WorldConstants


ROBOT_COLOR_STD = 0.15
ROBOT_COLOR_DECAY = 0.95
ENV_COLOR_STD = 0.15
ENV_COLOR_DECAY = 0.95
ROBOT_FINGER_KEY_1 = 'robot_finger_60_link_3'
ROBOT_FINGER_KEY_2 = 'robot_finger_120_link_3'
ROBOT_FINGER_KEY_3 = 'robot_finger_300_link_3'


def create_settings(seed=42, create_images=True):
    settings = {
        'seed': seed,
        'create_images': create_images,
        'skip_frame': 25,
        'policy': 'random',
        'action_mode': 'joint_positions'
    }
    np.random.seed(seed)
    return settings

def create_environment(settings):
    task = generate_task(task_generator_id='no_reward')
    env = CausalWorld(task=task,
                      observation_mode='pixel' if settings['create_images'] else 'structured',
                      action_mode=settings['action_mode'],
                      skip_frame=settings['skip_frame'],
                      seed=settings['seed'])
    env.add_ground_truth_state_to_info()
    env.set_intervention_space('space_a_b')
    _ = env.reset()
    env.do_intervention(
            {'gravity': np.array([0.0, 0.0, -9.81]),
             'goal_block': {'color': np.array([0.5, 1.0, 0.5])}}, check_bounds=False)
    return env

def color_to_np(color):
    if not isinstance(color, np.ndarray):
        color = np.array(color)
    return color

def mask_goal(obs):
    obs = obs * (obs[...,0:1] == obs[...,2:3]) * (np.abs(obs[...,0:1] * 2 - obs[...,1:2]) < 0.01)
    return obs

def combine_obs_and_diff(obs, diff):
    if diff.shape[-1] == 1:
        diff = np.concatenate([diff * (diff > 0), -diff * (diff < 0), diff * 0.], axis=-1)
    stacked_obs = np.stack([obs[:3], diff], axis=0)
    padded_obs = np.pad(stacked_obs, ((0, 0), (0, 0), (8, 8), (8, 8), (0, 0)))
    padded_obs = padded_obs.transpose(0, 2, 1, 3, 4)
    padded_obs = padded_obs.reshape(padded_obs.shape[0] * padded_obs.shape[1], -1, padded_obs.shape[-1])
    padded_obs = (padded_obs * 255).astype(np.uint8)
    return padded_obs

def inverse_sigmoid(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sample_robot_color(robot_id):
    color = np.array([0.6,
                      np.random.uniform(0, 1.0),
                      0.0])
    causal_var = color[1]
    color = np.roll(color, robot_id)
    return color, causal_var

def transition_robot_color(color, robot_id):
    color = np.roll(color, -robot_id)
    causal_var = inverse_sigmoid(color[1])
    causal_var = causal_var * ROBOT_COLOR_DECAY + np.random.normal(0, ROBOT_COLOR_STD)
    color[1] = sigmoid(causal_var)
    color = np.roll(color, robot_id)
    return color

def sample_floor_vars():
    factor = np.random.uniform(0, 1.0)
    floor_color = np.array([0.1, 0.1, 0.1]) + np.array([0.45, 0.45, 0.9]) * factor
    floor_friction = (1 - factor) * 0.3
    return floor_color, floor_friction, factor

def transition_floor_vars(floor_color, floor_friction):
    factor = 1 - (floor_friction / 0.3)
    causal_var = inverse_sigmoid(factor)
    causal_var = causal_var * ENV_COLOR_DECAY + np.random.normal(0, ENV_COLOR_STD)
    factor = sigmoid(causal_var)
    floor_color = np.array([0.1, 0.1, 0.1]) + np.array([0.45, 0.45, 0.9]) * factor
    floor_friction = (1 - factor) * 0.3
    return floor_color, floor_friction

def sample_stage_vars():
    factor = np.random.uniform(0, 1.0)
    stage_color = np.array([1.0, 1.0, 1.0]) * (0.5 * (factor + 1))
    stage_friction = (1 - factor) * 0.4 + 0.1
    return stage_color, stage_friction, factor

def transition_stage_vars(stage_color, stage_friction):
    factor = 1 - ((stage_friction - 0.1) / 0.4)
    causal_var = inverse_sigmoid(factor)
    causal_var = causal_var * ENV_COLOR_DECAY + np.random.normal(0, ENV_COLOR_STD)
    factor = sigmoid(causal_var)
    stage_color = np.array([1.0, 1.0, 1.0]) * (0.5 * (factor + 1))
    stage_friction = (1 - factor) * 0.4 + 0.1
    return stage_color, stage_friction

def sample_object_color():
    factor = np.random.uniform(0, 1.0)
    object_color = np.array([0.5, factor, 0.5])
    object_friction = (1 - factor) * 0.25 + 0.05
    return object_color, object_friction, factor

def transition_object_color(object_color, object_friction):
    factor = 1 - ((object_friction - 0.05) / 0.25)
    causal_var = inverse_sigmoid(factor)
    causal_var = causal_var * ENV_COLOR_DECAY + np.random.normal(0, ENV_COLOR_STD)
    factor = sigmoid(causal_var)
    object_color = np.array([0.5, factor, 0.5])
    object_friction = (1 - factor) * 0.25 + 0.05
    return object_color, object_friction

def take_environment_step(env):
    # Perform action
    env._robot.apply_action(env.action_space.sample())
    info = env._task.get_info()
    collisions = env._robot._last_collisions
    
    # Interventions
    interventions = []
    
    # Get variable information
    var_info = info['ground_truth_current_state_variables']
    joint_positions = var_info['joint_positions']
    robot0_color = color_to_np(var_info[ROBOT_FINGER_KEY_1]['color'])
    robot1_color = color_to_np(var_info[ROBOT_FINGER_KEY_2]['color'])
    robot2_color = color_to_np(var_info[ROBOT_FINGER_KEY_3]['color'])
    floor_color = color_to_np(var_info['floor_color'])
    floor_friction = var_info['floor_friction']
    stage_color = color_to_np(var_info['stage_color'])
    stage_friction = var_info['stage_friction']
    object_color = color_to_np(var_info['tool_block']['color'])
    object_friction = var_info['tool_block']['friction']
    
    # Idea: under collisions, the corresponding trifinger changes color on its tip
    actual_collisions = [c for c in collisions if len(c) > 0]
    if len(actual_collisions) > 0 and len(collisions[0]) == 0:
        interventions.append('object_pos')
        collision_objects = set([c[0][3] for c in actual_collisions])
        for object_id in collision_objects:
            if object_id == 4:
                robot0_color, _ = sample_robot_color(0)
                interventions.append('robot0_color')
            elif object_id == 9:
                robot1_color, _ = sample_robot_color(1)
                interventions.append('robot1_color')
            elif object_id == 14:
                robot2_color, _ = sample_robot_color(2)
                interventions.append('robot2_color')
    if 'robot0_color' not in interventions:
        robot0_color = transition_robot_color(robot0_color, 0)
    if 'robot1_color' not in interventions:
        robot1_color = transition_robot_color(robot1_color, 1)
    if 'robot2_color' not in interventions:
        robot2_color = transition_robot_color(robot2_color, 2)
    
    # Background color changes if fingers are below a certain height (3rd dimension for all). Affects the friction of the ground
    if any([joint_positions[i] < -1.8 for i in [2, 5, 8]]):
        floor_color, floor_friction, _ = sample_floor_vars()
        interventions.append('floor_color')
    else:
        floor_color, floor_friction = transition_floor_vars(floor_color, floor_friction)
    
    # Stage color changes if fingers are above a certain height (3rd dimension for all). Affects the friction of the stage
    if any([joint_positions[i] > 1.8 for i in [2, 5, 8]]):
        stage_color, stage_friction, _ = sample_stage_vars()
        interventions.append('stage_color')
    else:
        stage_color, stage_friction = transition_stage_vars(stage_color, stage_friction)
        
    # Object color changes if fingers are close together. Affects friction of object
    if all([joint_positions[i] > 0.15 for i in [1, 4, 7]]):
        object_color, object_friction, _ = sample_object_color()
        interventions.append('object_color')
    else:
        object_color, object_friction = transition_object_color(object_color, object_friction)
    
    # Perform full intervention on causal variables
    env.do_intervention(
            {'stage_color': stage_color,
             'floor_color': floor_color,
             'stage_friction': stage_friction,
             'floor_friction': floor_friction,
             ROBOT_FINGER_KEY_1:  {'color': robot0_color},
             ROBOT_FINGER_KEY_2: {'color': robot1_color},
             ROBOT_FINGER_KEY_3: {'color': robot2_color},
             'tool_block': {'color': object_color,
                            'friction': object_friction}}, 
        check_bounds=False, render=False)
    info = env._task.get_info()['ground_truth_current_state_variables']
    
    if env._observation_mode == 'pixel':
        # Render image
        obs = env._robot.get_current_camera_observations()
        diff_frame = get_diff_frame(env)
    else:
        obs = np.zeros((3, 128, 128, 3), dtype=np.float32)
        diff_frame = np.zeros((3, 128, 128, 1), dtype=np.float32)
    
    return obs, diff_frame, info, interventions

def get_diff_frame(env):
    var_info = env._task.get_info()['ground_truth_current_state_variables']

    # Get velocity diff image
    last_position = env._robot._last_object_positions['tool_block']['cylindrical_position']
    last_orientation = env._robot._last_object_positions['tool_block']['orientation']
    env.do_intervention({'goal_block': {'cylindrical_position': last_position,
                                        'orientation': last_orientation}}, check_bounds=False)
    env._stage.update_goal_image()
    frame1 = env._stage.get_current_goal_image()

    new_position = var_info['tool_block']['cylindrical_position']
    new_orientation = var_info['tool_block']['orientation']
    env.do_intervention({'goal_block': {'cylindrical_position': new_position,
                                        'orientation': new_orientation}}, check_bounds=False)
    env._stage.update_goal_image()
    frame2 = env._stage.get_current_goal_image()
    diff_frame = (mask_goal(frame2) - mask_goal(frame1))[...,1:2]
    return diff_frame

def sample_random_position(env=None, sample_joints=False, sample_obj_pos=True):
    causal_vars = {}
    joint_positions = np.zeros(9,)
    for i in range(9):
        vals = 1e6
        if i % 3 == 0:
            mean, std, minv, maxv = -0.29, 0.34, -0.8, 0.5
        elif i % 3 == 1:
            mean, std, minv, maxv = 0.15, 0.34, -0.5, 1.0
        elif i % 3 == 2:
            mean, std, minv, maxv = 0.0, 0.9, -2.0, 2.0
        while vals > maxv or vals < minv:
            vals = np.random.normal(loc=mean, scale=std)
        joint_positions[i] = vals
    joint_velocities = np.zeros(9,)

    robot0_color, causal_vars['robot0_color'] = sample_robot_color(0)
    robot1_color, causal_vars['robot1_color'] = sample_robot_color(1)
    robot2_color, causal_vars['robot2_color'] = sample_robot_color(2)

    floor_color, floor_friction, causal_vars['floor_color'] = sample_floor_vars()
    stage_color, stage_friction, causal_vars['stage_color'] = sample_stage_vars()
    object_color, object_friction, causal_vars['object_color'] = sample_object_color()

    object_position = np.array([
            np.random.uniform(0.0, 0.15),
            np.random.uniform(-np.pi, np.pi),
            0.065 / 2.0
        ])
    names = ['dist', 'angle', 'height']
    for i in range(object_position.shape[0]):
        causal_vars[f'object_position_{names[i]}'] = object_position[i]
    object_orientation = np.array([
            0, 0, np.random.uniform(0, np.pi / 2.)
        ])
    causal_vars['object_orientation_angle'] = object_orientation[-1] * 4.

    intervention_dict = {
        'joint_positions': joint_positions,
        'joint_velocities': joint_velocities,
        'stage_color': stage_color,
        'floor_color': floor_color,
        'stage_friction': stage_friction,
        'floor_friction': floor_friction,
        ROBOT_FINGER_KEY_1: {'color': robot0_color},
        ROBOT_FINGER_KEY_2: {'color': robot1_color},
        ROBOT_FINGER_KEY_3: {'color': robot2_color},
        'tool_block': {'color': object_color,
                       'friction': object_friction,
                       'cylindrical_position': object_position,
                       'euler_orientation': object_orientation,
                       'linear_velocity': np.zeros(3,),
                       'angular_velocity': np.zeros(3,)}
    }

    if not sample_joints:
        intervention_dict.pop('joint_positions')
        intervention_dict.pop('joint_velocities')
    if not sample_obj_pos:
        intervention_dict['tool_block'].pop('cylindrical_position')
        intervention_dict['tool_block'].pop('euler_orientation')
        intervention_dict['tool_block'].pop('linear_velocity')
        intervention_dict['tool_block'].pop('angular_velocity')
        causal_vars.pop('object_orientation_angle')
        causal_vars.pop('object_position_dist')
        causal_vars.pop('object_position_angle')
        causal_vars.pop('object_position_height')

    if env is not None:
        if not sample_joints:
            info = env._task.get_info()['ground_truth_current_state_variables']
            intervention_dict['joint_positions'] = info['joint_positions']
            intervention_dict['joint_velocities'] = info['joint_velocities']
        success_signal, _ = env.do_intervention(intervention_dict, check_bounds=False, render=False)
        client = env._robot._pybullet_client_full_id
        if client is None:
            client = env._robot._pybullet_client_w_o_goal_id
        pybullet.performCollisionDetection(physicsClientId=client)
        contact_points = (pybullet.getContactPoints(1, 4, 
                                                    physicsClientId=client))
        if (success_signal is not None and not success_signal) or len(contact_points) > 0:
            if not sample_joints:
                env._robot.apply_action(env.action_space.sample())
            intervention_dict, causal_vars = sample_random_position(env=env, sample_joints=sample_joints, sample_obj_pos=sample_obj_pos)
    if 'object_velocity' not in causal_vars:
        info = env._task.get_info()['ground_truth_current_state_variables']
        causal_vars['object_velocity'] = np.linalg.norm(info['tool_block']['linear_velocity'])
        if not sample_obj_pos:
            causal_vars['object_orientation_angle'] = euler_from_quaternion(*info['tool_block']['orientation'])[-1] * 4.
            causal_vars['object_position_dist'] = info['tool_block']['cylindrical_position'][0]
            causal_vars['object_position_angle'] = info['tool_block']['cylindrical_position'][1]
            causal_vars['object_position_height'] = info['tool_block']['cylindrical_position'][2]
    return intervention_dict, causal_vars

def euler_from_quaternion(x, y, z, w):
    """
    Full credit for this function to https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def sample_indep_observation(env, sample_joints=False, sample_obj_pos=True):
    env._robot.apply_action(env.action_space.sample())
    intervention_dict, causal_vars = sample_random_position(env, sample_joints=sample_joints, sample_obj_pos=sample_obj_pos)
    causal_vars['intervention_dict'] = intervention_dict

    if env._observation_mode == 'pixel':
        obs = env._robot.get_current_camera_observations()
        diff = get_diff_frame(env)
    else:
        obs = np.zeros((3, 128, 128, 3), dtype=np.float32)
        diff = np.zeros((3, 128, 128, 1), dtype=np.float32)
    return obs, diff, causal_vars

def create_nested_dict(orig_dict, num_elements):
    nested_dict = {}
    for key in orig_dict:
        if isinstance(orig_dict[key], (float, int)):
            nested_dict[key] = np.zeros((num_elements,), dtype=np.float32)
        elif isinstance(orig_dict[key], np.ndarray):
            nested_dict[key] = np.zeros((num_elements, *orig_dict[key].shape),
                                        dtype=orig_dict[key].dtype)
        elif isinstance(orig_dict[key], tuple):
            nested_dict[key] = np.zeros((num_elements, len(orig_dict[key])),
                                        dtype=np.float32)
        elif isinstance(orig_dict[key], dict):
            nested_dict[key] = create_nested_dict(orig_dict[key], num_elements)
        else:
            print('Not sure how to handle', key, orig_dict[key])
    return nested_dict

def add_element_to_nested_dict(nested_dict, single_dict, idx):
    for key in single_dict:
        if isinstance(nested_dict[key], dict):
            add_element_to_nested_dict(nested_dict[key], single_dict[key], idx)
        else:
            nested_dict[key][idx] = single_dict[key]

def flatten_dict(nested_dict):
    all_keys = list(nested_dict.keys())
    for key in all_keys:
        if isinstance(nested_dict[key], dict):
            sub_dict = nested_dict.pop(key)
            flatten_dict(sub_dict)
            for sub_key in sub_dict:
                nested_dict[f'{key}/{sub_key}'] = sub_dict[sub_key]
        else:
            pass
    return nested_dict

def unflatten_dict(flatten_dict):
    all_keys = list(flatten_dict.keys())
    for key in all_keys:
        if '/' in key:
            main_key, sub_key = key.split('/', 1)
            if main_key not in flatten_dict:
                flatten_dict[main_key] = dict()
            flatten_dict[main_key][sub_key] = flatten_dict.pop(key)
    for key in flatten_dict:
        if isinstance(flatten_dict[key], dict):
            unflatten_dict(flatten_dict[key])
    return flatten_dict

def optimize_images_batchwise(observations, diffs, batch_size):
    start_idx = 0
    image_buffer = None
    while start_idx < observations.shape[0]:
        optimized_images = optimize_images(observations[start_idx:start_idx+batch_size], 
                                           diffs[start_idx:start_idx+batch_size] if diffs is not None else None)
        if image_buffer is None:
            image_buffer = np.zeros((observations.shape[0], *optimized_images.shape[1:]),
                                    dtype=optimized_images.dtype)
        image_buffer[start_idx:start_idx+batch_size] = optimized_images
        del optimized_images
        start_idx += batch_size
    return image_buffer

def optimize_images(observations, diffs):
    if diffs is None:
        diffs = np.zeros((*observations.shape[:-1], 1), dtype=np.float32)
    if diffs.dtype == np.int32:
        diffs = diffs.astype(np.float32) / 255.
    if diffs.dtype == np.float64:
        diffs = diffs.astype(np.float32)
    if diffs.dtype == np.float32:
        diffs = (diffs + 1.) / 2.
    if observations.dtype == np.uint8:
        observations = observations.astype(np.float32) / 255.
    if observations.dtype == np.float64:
        observations = observations.astype(np.float32)
    images = np.concatenate([observations, diffs], axis=-1)
    # Bring channels in front of height and width
    images = np.moveaxis(images, -1, -3)
    # Flatten channel and number of perspectives (3)
    images = np.reshape(images, images.shape[:-4] + (-1,) + images.shape[-2:])
    # Resizing to 64x64
    images = torch.from_numpy(images)
    if len(observations.shape) == 6:
        images = images.flatten(0, 1)
    images = F.interpolate(images.float(), 
                           scale_factor=(0.5, 0.5),
                           mode='bilinear')
    if len(observations.shape) == 6:
        images = images.unflatten(0, observations.shape[:2])
    images = images.numpy()
    # To numpy uint8
    images = (images * 255.).astype(np.uint8)
    # Return
    return images

def generate_sequence(env, settings, seq_len, save_dir):
    _ = sample_random_position(env)
    obs, diff_frame, info, intv = take_environment_step(env)

    all_observations = np.zeros((seq_len, *obs.shape), dtype=obs.dtype)
    all_diffs = np.zeros((seq_len, *diff_frame.shape), dtype=diff_frame.dtype)
    all_infos = create_nested_dict(info, num_elements=seq_len)
    all_interventions = defaultdict(lambda: np.zeros((seq_len,), dtype=np.uint8))
    for key in ['robot0_color', 'robot1_color', 'robot2_color',
                'floor_color', 'stage_color', 'object_color',
                'object_pos']:
        all_interventions[key][0] = 0

    def log(obs, diff_frame, info, intv, idx):
        all_observations[idx] = obs
        all_diffs[idx] = diff_frame
        add_element_to_nested_dict(all_infos, info, idx=idx)
        for var_key in intv:
            all_interventions[var_key][idx] = 1

    log(obs, diff_frame, info, intv, 0)

    for idx in (range(1, seq_len)):
        obs, diff_frame, info, intv = take_environment_step(env)
        log(obs, diff_frame, info, intv, idx)

    optimized_images = optimize_images_batchwise(all_observations, all_diffs,
                                                 batch_size=250)
    all_observations = (all_observations * 255.).astype(np.uint8)
    all_diffs = (all_diffs * 255.).astype(np.int32)

    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, 'optimized_images.npz'),
                        imgs=optimized_images)
    np.savez_compressed(os.path.join(save_dir, 'observations.npz'),
                        obs=all_observations)
    np.savez_compressed(os.path.join(save_dir, 'diff_frames.npz'),
                        diff=all_diffs)
    np.savez_compressed(os.path.join(save_dir, 'step_infos.npz'),
                        **flatten_dict(all_infos))
    np.savez_compressed(os.path.join(save_dir, 'interventions.npz'),
                        **flatten_dict(all_interventions))

    if save_dir.endswith('train/seq_0000/'):
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
        max_images = 100
        observations = all_observations[:max_images]
        diffs = all_diffs[:max_images]
        observations = observations.transpose(0, 2, 1, 3, 4)
        diffs = diffs.transpose(0, 2, 1, 3, 4)
        diffs = np.concatenate([diffs * (diffs > 0), -diffs * (diffs < 0), diffs * 0], axis=-1).astype(np.uint8)
        observations = np.stack([observations, diffs], axis=1)
        observations = np.pad(observations, ((0, 0), (0, 0), (4, 4), (0, 0), (4, 4), (0, 0)), mode='constant', constant_values=128)
        observations = observations.reshape(observations.shape[0], observations.shape[1] * observations.shape[2], observations.shape[3] * observations.shape[4], observations.shape[5])
        observations = np.pad(observations, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='constant', constant_values=128)
        for i in range(observations.shape[0]):
            imwrite(os.path.join(save_dir, 'images', f'obs_{i:04d}.png'), observations[i])

def generate_indep_set(env, settings, set_size, save_dir):
    obs, diff, causal_vars = sample_indep_observation(env)
    all_observations = np.zeros((set_size, *obs.shape), dtype=obs.dtype)
    all_diffs = np.zeros((set_size, *diff.shape), dtype=diff.dtype)
    all_causal_vars = create_nested_dict(causal_vars, num_elements=set_size)

    def log(obs, diff, causal_vars, idx):
        all_observations[idx] = obs
        all_diffs[idx] = diff
        add_element_to_nested_dict(all_causal_vars, causal_vars, idx=idx)

    log(obs, diff, causal_vars, 0)

    for idx in range(1, set_size):
        env._robot.apply_action(env.action_space.sample())
        obs, diff, causal_vars = sample_indep_observation(env, sample_obj_pos=(idx % 10 == 0))
        log(obs, diff, causal_vars, idx)

    all_observations = (all_observations * 255.).astype(np.uint8)
    all_diffs = (all_diffs * 255.).astype(np.int32)
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, 'observations.npz'),
                        obs=all_observations)
    np.savez_compressed(os.path.join(save_dir, 'diff_frames.npz'),
                        diff=all_diffs)
    np.savez_compressed(os.path.join(save_dir, 'causal_vars.npz'),
                        **flatten_dict(all_causal_vars))

def generate_specified_set(env, settings, intervention_sets, save_dir):
    all_observations = None
    for idx, intv_set in enumerate(intervention_sets):
        env.do_intervention(intv_set, check_bounds=False)
        if env._observation_mode == 'pixel':
            obs = env._robot.get_current_camera_observations()
        else:
            obs = np.zeros((3, 128, 128, 3), dtype=np.float32)
        if all_observations is None:
            all_observations = np.zeros((len(intervention_sets), *obs.shape), dtype=obs.dtype)
        all_observations[idx] = obs
    all_observations = (all_observations * 255.).astype(np.uint8)
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, 'observations.npz'),
                        obs=all_observations)

def start_sequence_generation(inputs):
    seed, seq_idx, seq_len, extra_kwargs, save_dir = inputs
    settings = create_settings(seed=seed, **extra_kwargs)
    env = create_environment(settings)
    generate_sequence(env, settings, seq_len=seq_len, save_dir=os.path.join(save_dir, f'seq_{str(seq_idx).zfill(4)}/'))
    env.close()

def start_indep_generation(inputs):
    seed, set_idx, set_size, extra_kwargs, save_dir = inputs
    settings = create_settings(seed=seed, **extra_kwargs)
    env = create_environment(settings)
    generate_indep_set(env, settings, set_size=set_size, save_dir=os.path.join(save_dir, f'set_{str(set_idx).zfill(4)}/'))
    env.close()

def start_specified_image_generation(inputs):
    extra_kwargs, set_idx, intervention_sets, save_dir = inputs
    settings = create_settings(seed=0, **extra_kwargs)
    env = create_environment(settings)
    generate_specified_set(env, settings, intervention_sets, save_dir=os.path.join(save_dir, f'spec_{str(set_idx).zfill(4)}/'))
    env.close()

def time_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        duration = int(time.time() - start_time)
        print(f'It took {duration//60} minutes {duration%60:02d} seconds')
    return wrapper

@time_func
def generate_train_sequences(args):
    print('Starting generation of training dataset...')
    extra_kwargs = {'create_images': not args.skip_images}
    train_folder = os.path.join(args.output_folder, 'train/')
    os.makedirs(train_folder, exist_ok=True)
    process_inputs = zip(list(range(args.seed, args.seed + args.num_sequences)),
                         list(range(args.num_sequences)),
                         [args.seq_len] * args.num_sequences,
                         [extra_kwargs] * args.num_sequences,
                         [train_folder] * args.num_sequences)
    with Pool(args.num_processes) as p:
        p.map(start_sequence_generation, process_inputs)

@time_func
def generate_indep_samples(args, subset='val'):
    print(f'Starting generation of {subset}_indep dataset...')
    extra_kwargs = {'create_images': not args.skip_images}
    subset_folder = os.path.join(args.output_folder, f'{subset}_indep/')
    os.makedirs(subset_folder, exist_ok=True)
    base_seed = args.seed + args.num_sequences + sum([ord(c) for c in subset])
    np.random.seed(base_seed)
    base_seed = np.random.randint(base_seed, high=int(1e8))
    process_inputs = zip(list(range(base_seed, 
                                    base_seed + args.num_indep_sets)),
                         list(range(args.num_indep_sets)),
                         [args.seq_len] * args.num_indep_sets,
                         [extra_kwargs] * args.num_indep_sets,
                         [subset_folder] * args.num_indep_sets)
    with Pool(args.num_processes) as p:
        p.map(start_indep_generation, process_inputs)
    all_folders = sorted(glob(os.path.join(subset_folder, 'set_*/')))
    all_observations = [np.load(os.path.join(f, 'observations.npz'))['obs'] for f in all_folders]
    all_observations = np.concatenate(all_observations, axis=0)
    all_diffs = [np.load(os.path.join(f, 'diff_frames.npz'))['diff'] for f in all_folders]
    all_diffs = np.concatenate(all_diffs, axis=0)
    optimized_images = optimize_images_batchwise(all_observations, all_diffs,
                                                 batch_size=250)
    np.savez_compressed(os.path.join(subset_folder, 'optimized_images.npz'), imgs=optimized_images)
    np.savez_compressed(os.path.join(subset_folder, 'observations.npz'), obs=all_observations)
    np.savez_compressed(os.path.join(subset_folder, 'diff_frames.npz'), diff=all_diffs)
    all_causal_vars = [np.load(os.path.join(f, 'causal_vars.npz')) for f in all_folders]
    all_causal_vars = {key: np.concatenate([c[key] for c in all_causal_vars], axis=0) for key in all_causal_vars[0]}
    np.savez_compressed(os.path.join(subset_folder, 'causal_vars.npz'), **all_causal_vars)
    for folder in all_folders:
        os.remove(os.path.join(folder, 'observations.npz'))
        os.remove(os.path.join(folder, 'diff_frames.npz'))
        os.remove(os.path.join(folder, 'causal_vars.npz'))
        os.rmdir(folder)
    
@time_func
def generate_triplet_samples(args, subset='val'):
    print(f'Starting generation of {subset}_triplets dataset...')
    # Step 1: Generate novel sequences
    extra_kwargs = {'create_images': not args.skip_images}
    subset_folder = os.path.join(args.output_folder, f'{subset}_triplets/')
    os.makedirs(subset_folder, exist_ok=True)
    base_seed = args.seed + args.num_sequences + args.num_indep_sets + sum([ord(c) for c in subset])
    np.random.seed(base_seed)
    base_seed = np.random.randint(base_seed, high=int(1e8))
    process_inputs = zip(list(range(base_seed, 
                                    base_seed + args.num_triplet_sets)),
                         list(range(args.num_triplet_sets)),
                         [args.seq_len] * args.num_triplet_sets,
                         [extra_kwargs] * args.num_triplet_sets,
                         [subset_folder] * args.num_triplet_sets)
    print('Generating sequences...')
    with Pool(args.num_processes) as p:
        p.map(start_sequence_generation, process_inputs)
    print('Done')
    # Step 2: Load sequences to one dataset
    all_folders = sorted(glob(os.path.join(subset_folder, 'seq_*/')))
    all_observations = np.stack(
            [np.load(os.path.join(f, 'observations.npz'))['obs'] 
             for f in all_folders], axis=0
        )
    all_diffs = np.stack(
            [np.load(os.path.join(f, 'diff_frames.npz'))['diff'] 
             for f in all_folders], axis=0
        )
    all_infos = [np.load(os.path.join(f, 'step_infos.npz')) for f in all_folders]
    step_infos = {key: np.stack([c[key] for c in all_infos], axis=0) for key in all_infos[0]}
    all_interventions = [np.load(os.path.join(f, 'interventions.npz')) for f in all_folders]
    all_interventions = {key: np.stack([c[key] for c in all_interventions], axis=0) for key in all_interventions[0]}
    # Step 2.1: Save these sequences as normal val/test dataset
    indv_folder = os.path.join(args.output_folder, f'{subset}/')
    os.makedirs(indv_folder, exist_ok=True)
    optimized_images = optimize_images(all_observations, all_diffs)
    np.savez_compressed(os.path.join(indv_folder, 'optimized_images.npz'), imgs=optimized_images)
    del optimized_images
    np.savez_compressed(os.path.join(indv_folder, 'observations.npz'), obs=all_observations)
    np.savez_compressed(os.path.join(indv_folder, 'diff_frames.npz'), diff=all_diffs)
    np.savez_compressed(os.path.join(indv_folder, 'interventions.npz'), **all_interventions)
    np.savez_compressed(os.path.join(indv_folder, 'step_infos.npz'), **step_infos)
    # Step 2.2: Prepare infos as list of dicts for easier handling
    all_observations = np.reshape(all_observations, (-1, *all_observations.shape[2:]))
    all_diffs = np.reshape(all_diffs, (-1, *all_diffs.shape[2:]))
    all_infos = [{key: info[key] for key in info} for info in all_infos]
    all_infos = [[{key: info[key][i] for key in info} for i in range(info[next(iter(info.keys()))].shape[0])]
                 for info in all_infos]
    all_infos = [sub_info for info in all_infos for sub_info in info]
    # Step 3: Create triplet pairs with needed info dicts
    pair_idxs = np.random.permutation(all_observations.shape[0])
    sets_of_vars = [('robot', ['joint_positions']),
                    ('stage_color', ['stage_color', 'stage_friction']),
                    ('floor_color', ['floor_color', 'floor_friction']),
                    ('robot0_color', [f'{ROBOT_FINGER_KEY_1}/color']),
                    ('robot1_color', [f'{ROBOT_FINGER_KEY_2}/color']),
                    ('robot2_color', [f'{ROBOT_FINGER_KEY_3}/color']),
                    ('object_color', ['tool_block/color', 'tool_block/friction']),
                    ('object_pos', ['tool_block/cylindrical_position', 'tool_block/orientation', 
                                    'tool_block/linear_velocity', 'tool_block/angular_velocity'])]
    triplet_masks = np.zeros((pair_idxs.shape[0], len(sets_of_vars)), dtype=np.float32)
    get_replace_mask = lambda arr: (np.logical_or(arr.sum(axis=-1) == 0, arr.sum(axis=-1) == arr.shape[-1]))
    while get_replace_mask(triplet_masks).any():
        new_triplets = np.random.binomial(n=1, p=0.5, size=(pair_idxs.shape[0], len(sets_of_vars)))
        triplet_masks = np.where(get_replace_mask(triplet_masks)[...,None], new_triplets, triplet_masks)
    new_infos = [{key: all_infos[i if triplet_masks[i,key_idx] == 0 else pair_idxs[i]][key] for key_idx, key_list in enumerate(sets_of_vars) for key in key_list[1]}
                 for i in range(pair_idxs.shape[0])]
    unflattened_infos = [unflatten_dict(info) for info in new_infos]

    start_time = time.time()
    settings = create_settings(seed=np.random.randint(0, int(1e8)), 
                               create_images=False)
    env = create_environment(settings)
    for i, intv_set in enumerate(unflattened_infos):
        env.do_intervention(intv_set, check_bounds=False)
        client = env._robot._pybullet_client_full_id
        if client is None:
            client = env._robot._pybullet_client_w_o_goal_id
        pybullet.performCollisionDetection(physicsClientId=client)
        contact_points = (pybullet.getContactPoints(1, 4, 
                                                    physicsClientId=client))
        if len(contact_points) > 0:
            triplet_masks[i,0] = 0
            triplet_masks[i,-1] = 0
            for key in (sets_of_vars[0][1] + sets_of_vars[-1][1]):
                new_infos[i][key] = all_infos[i][key]
            unflattened_infos[i] = unflatten_dict(new_infos[i])
    end_time = time.time()
    print(f'Time to check faulty triplets: {end_time - start_time:4.2f} seconds')
    # Step 4: Run processes in parallel for generating the 'third' images
    pidxs = np.linspace(0, len(unflattened_infos), num=args.num_processes+1).astype(np.int32)
    process_inputs = zip([extra_kwargs] * args.num_processes,
                         list(range(args.num_processes)),
                         [unflattened_infos[pidxs[i]:pidxs[i+1]] for i in range(args.num_processes)],
                         [subset_folder] * args.num_processes)
    print('Generating triplet images...')
    with Pool(args.num_processes) as p:
        p.map(start_specified_image_generation, process_inputs)
    print('Done')
    # Step 5: Combine everything together in single dataset
    triplet_folders = sorted(glob(os.path.join(subset_folder, 'spec_*/')))
    triplet_observations = np.concatenate(
            [np.load(os.path.join(f, 'observations.npz'))['obs'] 
             for f in triplet_folders], axis=0
        )
    final_observations = np.stack([all_observations,
                                   all_observations[pair_idxs],
                                   triplet_observations], axis=1)
    final_diffs = np.stack([all_diffs,
                            all_diffs[pair_idxs],
                            all_diffs[np.where(triplet_masks[:,-1] == 1,
                                               pair_idxs,
                                               np.arange(pair_idxs.shape[0], dtype=np.int32))]], axis=1)
    final_infos = [(all_infos[i], 
                    all_infos[pair_idxs[i]],
                    flatten_dict(new_infos[i])) for i in range(len(all_infos))]
    final_infos = [{key: np.stack([inf[key] for inf in infos], axis=0) for key in infos[-1]}
                    for infos in final_infos]
    final_infos = {key: np.stack([inf[key] for inf in final_infos], axis=0) for key in final_infos[0]}

    optimized_images = optimize_images_batchwise(final_observations, final_diffs,
                                                 batch_size=250)
    np.savez_compressed(os.path.join(subset_folder, 'optimized_images.npz'), imgs=optimized_images)
    np.savez_compressed(os.path.join(subset_folder, 'observations.npz'), obs=final_observations)
    np.savez_compressed(os.path.join(subset_folder, 'diff_frames.npz'), diff=final_diffs)
    np.savez_compressed(os.path.join(subset_folder, 'infos.npz'), **final_infos)
    np.savez_compressed(os.path.join(subset_folder, 'triplet_masks.npz'), mask=triplet_masks)
    with open(os.path.join(subset_folder, 'causal_var_list.json'), 'w') as f:
        json.dump(sets_of_vars, f, indent=4)
    for folder in all_folders + triplet_folders:
        os.remove(os.path.join(folder, 'observations.npz'))
        if '/seq_' in folder:
            os.remove(os.path.join(folder, 'optimized_images.npz'))
            os.remove(os.path.join(folder, 'diff_frames.npz'))
            os.remove(os.path.join(folder, 'step_infos.npz'))
            os.remove(os.path.join(folder, 'interventions.npz'))
        os.rmdir(folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Folder to save the dataset to.')
    parser.add_argument('--num_sequences', type=int, default=100,
                        help='Number of sequences to generate from this dataset.')
    parser.add_argument('--seq_len', type=int, default=1000,
                        help='Number of frames to generate per sequence.')
    parser.add_argument('--num_indep_sets', type=int, default=25,
                        help='Number of sets of frames, each seq_len large, to generate for indep datasets.')
    parser.add_argument('--num_triplet_sets', type=int, default=10,
                        help='Number of sets of frames, each seq_len large, to generate for triplet datasets.')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes to use for the rendering.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Start seed for reproducibility.')
    parser.add_argument('--skip_images', action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    generate_train_sequences(args)
    if args.num_triplet_sets > 0:
        generate_triplet_samples(args, 'val')
        generate_triplet_samples(args, 'test')
    if args.num_indep_sets > 0:
        generate_indep_samples(args, 'train')
        generate_indep_samples(args, 'val')
        generate_indep_samples(args, 'test')