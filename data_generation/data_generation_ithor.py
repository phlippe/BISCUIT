""" 
Dataset generation of the iTHOR environment
"""

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from copy import deepcopy
from random import shuffle
import random
import numpy as np
from matplotlib import pyplot as plt
import imageio
import os
import torch
import torch.nn.functional as F
from collections import defaultdict
import json
import time
from hashlib import sha256
from scipy import signal

# The simulator itself runs on 512x512, but the images are downsampled by factor 2 afterward (256x256)
RESOLUTION = 512  
SIMPLE_SET = False
OBJECT_NAMES = [
    'Stove',
    'Microwave',
    'Cabinet',
    'Toaster',
    'Counter'
] + ([
    'Egg',
    'Pan',
    'Plate',
    'Potato'
] if not SIMPLE_SET else [])
MIN_DIST = 0.22

NOT_MOVABLE = [
    'Window',
    'Stove',
    'Sink',
    'Shelf',
    'LightSwitch'
]

FIXED_POSITION_DICT = [
    {
        "objectName": "Toaster",
        "rotation": {
            "x": 0, "y": 270, "z": 0
        },
        "position": {
            "x": 0.98, "y": 0.98, "z": -1.75
        }
    }
]
MOVABLE_POSITION_DICT = [
    {
        "objectName": "Egg",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        }
    },
    {
        "objectName": "Potato",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        }
    },
    {
        "objectName": "Plate",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        }
    }
]
COUNTER_POSITIONS = [
    {
        "objectName": None,
        "position": {
            "x": 0.75, "y": 0.98, "z": -0.35
        }
    },
    {
        "objectName": None,
        "position": {
            "x": 1.03, "y": 0.98, "z": -0.35
        }
    },
    {
        "objectName": None,
        "position": {
            "x": 0.65, "y": 0.98, "z": -0.55
        }
    }
]
CATEGORICAL_POSITION_DICT = [
    {
        "objectName": "Pan",
        "rotation": {
            "x": 0, "y": 0, "z": 0
        },
        "position": [
            {"x": 0.85, "y": 0.95, "z": -1.20}
        ]
    }
]
ACTIONS = {
    'PickupObject': [

    ] + (['Egg', 'Plate'] if not SIMPLE_SET else []),
    'PutObject': (['Pan'] if not SIMPLE_SET else [])  + 
    [
        'Microwave', 'CounterTop_f8092513'
    ],
    'ToggleObject': [
        'Toaster', 'Microwave', 'StoveKnob_38c1dbc2', 'StoveKnob_690d0d5d', 'StoveKnob_c8955f66', 'StoveKnob_cf670576'
    ],
    'SliceObject': ([
        'Potato'
    ] if not SIMPLE_SET else []),
    'OpenObject': [
        'Microwave', 'Cabinet_47fc321b'
    ],
    'NoOp': [
        'NoObject1', 'NoObject2', 'NoObject3'
    ]
}
INTERACT_OBJS = list(set([obj for action_key in ['PickupObject', 'ToggleObject', 'SliceObject', 'OpenObject']
                              for obj in ACTIONS[action_key]]))
PICKUP = {'position': None}



def move_objects_into_position(controller, event, positions):
    object_poses = [{key: obj[key] for key in ["name", "rotation", "position"]} 
                    for obj in event.metadata['objects'] if not any([obj['name'].startswith(n) for n in NOT_MOVABLE])]
    for obj in object_poses:
        obj['objectName'] = obj.pop('name')
    for position in positions:
        for obj in object_poses:
            if obj["objectName"].startswith(position["objectName"]):
                for key in ["position", "rotation"]:
                    obj[key] = position[key]
    event = controller.step(action='SetObjectPoses',
                            objectPoses=object_poses)
    return event


def initialize_environment(seed:int=42):
    np.random.seed(seed)
    random.seed(seed)

    controller = Controller(width=RESOLUTION, 
                            height=RESOLUTION, 
                            gridSize=0.1, 
                            platform=CloudRendering,
                            renderInstanceSegmentation=True)

    # Move the agent to the starting position
    event = controller.step(action="Stand")
    event = controller.step(action="MoveAhead")
    for _ in range(3):
        event = controller.step(action="MoveLeft")
    event = controller.step(action='LookDown', degrees=10)

    # Move the objects into position
    position_dict = deepcopy(FIXED_POSITION_DICT)
    movable_position_dict = MOVABLE_POSITION_DICT
    success_positions = False
    while not success_positions:
        success_positions = True
        for i, p in enumerate(COUNTER_POSITIONS):
            pos_found = False
            num_tries = 0
            while not pos_found and num_tries < 10:
                pos_found = True
                num_tries += 1
                x = np.random.uniform(0.65, 1.00)
                z = np.random.uniform(-0.35, -0.81)
                for p_alt in COUNTER_POSITIONS[:i]:
                    if np.abs(x - p_alt['position']['x']) < MIN_DIST and np.abs(z - p_alt['position']['z']) < MIN_DIST:
                        pos_found = False
                        break
            if pos_found:
                p['position']['x'] = x
                p['position']['z'] = z
            else:
                success_positions = False
                break
    assert all([pos['objectName'] is None for pos in COUNTER_POSITIONS]), str(COUNTER_POSITIONS)
    for mov_pos, count_pos in zip(movable_position_dict, COUNTER_POSITIONS):
        count_pos['objectName'] = mov_pos['objectName']
        mov_pos['position'] = count_pos['position']
        mov_pos['counter_position'] = count_pos
    position_dict.extend(deepcopy(movable_position_dict))
    categorical_position_dict = deepcopy(CATEGORICAL_POSITION_DICT)
    for pos in categorical_position_dict:
        pos['position'] = pos['position'][np.random.randint(0, len(pos['position']))]
    position_dict.extend(categorical_position_dict)
    event = move_objects_into_position(controller, event, position_dict)

    # Removing unnecessary objects
    objects = list(event.metadata['objects'])
    for obj in objects:
        if not any([obj['name'].startswith(name) for name in OBJECT_NAMES]):
            event = controller.step(action="DisableObject",
                                    objectId=obj['objectId'])

    # Place potato on plate
    if not SIMPLE_SET:
        for obj in objects:
            if obj['name'].startswith('Potato'):
                event = controller.step(action="PickupObject",
                                        objectId=obj['objectId'])
                break
        for obj in objects:
            if obj['name'].startswith('Plate'):
                event = controller.step(action="PutObject",
                                        objectId=obj['objectId'])
                break
        for mov_pos in MOVABLE_POSITION_DICT:
            if mov_pos['objectName'] == 'Potato':
                MOVABLE_POSITION_DICT.remove(mov_pos)
                break

    return controller, event


def get_environment_state(event : dict):
    state = {}
    for obj in event.metadata['objects']:
        object_name = obj['name']
        if not obj['visible'] or not any([object_name.startswith(name) for name in INTERACT_OBJS]):
            continue
        if obj['pickupable']:
            state[object_name + '_pickedup'] = int(obj['isPickedUp'])
            for key in obj['position']:
                state[object_name + '_center_' + key] = obj['position'][key]
                if obj['position'][key] == 0:
                    print('Suspicious position', object_name, obj['position'])
        if obj['openable']:
            state[object_name + '_open'] = int(obj['isOpen'])
        if obj['toggleable']:
            state[object_name + '_on'] = int(obj['isToggled'])
        if obj['sliceable'] and not obj['breakable'] and 'Knife' in OBJECT_NAMES:
            state[object_name + '_sliced'] = int(obj['isSliced'])
        if obj['breakable'] and object_name.startswith('Egg'):
            state[object_name + '_broken'] = int(obj['isBroken'])
        if obj['cookable']:
            state[object_name + '_cooked'] = int(obj['isCooked'])
    return state


def get_object_id(event, object_name):
    for obj in event.metadata['objects']:
        if obj['name'].startswith(object_name):
            return obj['objectId'], obj
    return None, None


def get_object_segmentations(event : dict):
    segmentations = {}
    for obj_name in INTERACT_OBJS:
        loc_objId, _ = get_object_id(event, obj_name)
        if not loc_objId in event.instance_masks:
            continue
        loc_mask = event.instance_masks[loc_objId]
        segmentations[obj_name] = loc_mask
    return segmentations


def get_action_pos(event : dict, objectId : str, action_type : str, randomize : bool = True, obj : dict = None):
    if action_type == 'PutObject':
        for sub_obj in event.metadata['objects']:
            if sub_obj['isPickedUp']:
                objectId = sub_obj['objectId']
                break
    if action_type == 'NoOp':
        instance_mask = None
        for obj_name in INTERACT_OBJS:
            loc_objId, _ = get_object_id(event, obj_name)
            if not loc_objId in event.instance_masks:
                continue
            loc_mask = event.instance_masks[loc_objId]
            if instance_mask is None:
                instance_mask = loc_mask
            else:
                instance_mask = instance_mask + loc_mask
        instance_mask = signal.convolve2d(instance_mask, np.ones((9, 9)), mode='same', boundary='fill', fillvalue=0)
        object_positions = np.where(instance_mask == 0)
    elif objectId not in event.instance_masks:
        print('Invalid action, try again')
        return None
    else:
        instance_mask = event.instance_masks[objectId]
        object_positions = np.where(instance_mask)
        # For the Microwave, we want to split the action into two parts
        # One toggles the microwave on/off, the other opens the door
        if objectId.startswith('Microwave'):
            q_high = np.percentile(object_positions[1], q=80)
            q_low = np.percentile(object_positions[1], q=30)
            if action_type == 'ToggleObject':
                pos_mask = object_positions[1] > q_high
            elif action_type == 'OpenObject' and not obj['isOpen']:
                pos_mask = object_positions[1] <= q_high
            elif action_type == 'OpenObject' and obj['isOpen']:
                pos_mask = object_positions[1] <= q_low
            elif action_type == 'PutObject':
                pos_mask = (object_positions[1] > q_low) * (object_positions[1] < q_high)
            else:
                print('Invalid action, try again')
                return None
            object_positions = tuple(pos[pos_mask] for pos in object_positions)
    if randomize:
        pos = np.random.randint(0, object_positions[0].shape[0])
    else:
        pos = 0
    object_position = [p[pos] for p in object_positions]
    action_pos = np.zeros(2, dtype=np.float32)
    for i in range(action_pos.shape[0]):
        action_pos[i] = object_position[i] * 1. / instance_mask.shape[i]
    return action_pos


def perform_action(controller : Controller, action_type : str, object_name : str, event : dict, step_number : int = -1):
    print((f"[{step_number:3d}] " if step_number >= 0 else "") + f"Performing action {action_type} on {object_name}")
    objectId, obj = get_object_id(event, object_name)
    action_pos = get_action_pos(event, objectId, action_type, obj=obj)
    if action_pos is not None:
        if action_type == 'PickupObject':
            event = controller.step(action='PickupObject',
                                    objectId=objectId)
            orig_pos = None
            for mov_pos in MOVABLE_POSITION_DICT:
                if object_name.startswith(mov_pos['objectName']):
                    if mov_pos['counter_position'] is not None:
                        orig_pos = mov_pos['counter_position']['position']
                        mov_pos['counter_position']['objectName'] = None
                        mov_pos['counter_position'] = None
                    break
            if orig_pos is None:
                PICKUP['position'] = (np.random.uniform(-0.15, 0.15), np.random.uniform(0.08, 0.15))
            else:
                right_init = ((orig_pos['z'] + 0.81) / (0.81 - 0.35)) * 0.3 - 0.15
                up_init = ((orig_pos['x'] - 0.65) / (1.00 - 0.65)) * 0.07 + 0.08
                PICKUP['position'] = (np.clip(-right_init + np.random.rand() * 0.05, -0.15, 0.15),
                                      np.clip(up_init + np.random.rand() * 0.02, 0.08, 0.15))
            event = controller.step(action="MoveHeldObject",
                                    ahead=0.0,
                                    right=PICKUP['position'][0],
                                    up=PICKUP['position'][1],
                                    forceVisible=False)
        elif action_type == 'PutObject':
            picked_object = None
            for sub_obj in event.metadata['objects']:
                if sub_obj['isPickedUp']:
                    picked_object = sub_obj
                    break
            if picked_object is None:
                print('No object picked up')
                return event, None
            event = controller.step(action='PutObject',
                                    objectId=objectId)
            PICKUP['position'] = None
            if object_name.startswith('CounterTop'):
                event, _ = perform_action(controller, action_type='MoveObject', object_name=picked_object['name'], event=event, step_number=step_number)
            elif object_name.startswith('Microwave'):
                pass
        elif action_type == 'MoveObject':
            orig_pos = None
            for mov_pos in MOVABLE_POSITION_DICT:
                if object_name.startswith(mov_pos['objectName']):
                    orig_pos = mov_pos
                    break
            pos_found = False
            while not pos_found:
                pos_found = True
                if orig_pos['counter_position'] is not None:
                    x = orig_pos['counter_position']['position']['x'] + np.random.rand() * 0.2
                    x = np.clip(x, 0.65, 1.00)
                    z = orig_pos['counter_position']['position']['z'] + np.random.rand() * 0.2
                    z = np.clip(z, -0.81, -0.35)
                else:
                    x = np.random.uniform(0.65, 1.00)
                    z = np.random.uniform(-0.35, -0.81)
                for mov_pos in MOVABLE_POSITION_DICT:
                    if not object_name.startswith(mov_pos['objectName']):
                        if mov_pos['counter_position'] is None:
                            continue
                        else:
                            if abs(x - mov_pos['counter_position']['position']['x']) < MIN_DIST and \
                                    abs(z - mov_pos['counter_position']['position']['z']) < MIN_DIST:
                                pos_found = False
                                break
            
            for mov_pos in MOVABLE_POSITION_DICT:
                if object_name.startswith(mov_pos['objectName']):
                    mov_pos['counter_position'] = {'position': {'x': x, 'y': 0.98, 'z': z}, 'objectName': object_name}
                    event = controller.step(
                        action="PlaceObjectAtPoint",
                        objectId=objectId,
                        position=mov_pos['counter_position']['position'],
                        rotation=mov_pos['rotation']
                    )
                    break
            
            if PICKUP['position'] is not None:
                event = controller.step(action="MoveHeldObject",
                                        ahead=0.0,
                                        right=PICKUP['position'][0],
                                        up=PICKUP['position'][1],
                                        forceVisible=False)
        elif action_type == 'ToggleObject':
            event = controller.step(action='ToggleObjectOn' if not obj['isToggled'] else 'ToggleObjectOff',
                                    objectId=objectId)
        elif action_type == 'SliceObject':
            event = controller.step(action='SliceObject',
                                    objectId=objectId)
            if object_name in ACTIONS['PickupObject']:
                ACTIONS['PickupObject'].remove(object_name)
            ACTIONS['SliceObject'].remove(object_name)
        elif action_type == 'OpenObject':
            if not obj['isOpen']:
                event = controller.step(action='OpenObject',
                                        objectId=objectId,
                                        openness=1.0)
            else:
                event = controller.step(action='CloseObject',
                                        objectId=objectId)
        elif action_type == 'NoOp':
            event = controller.step(action='Stand')
    return event, action_pos


def perform_random_action(controller : Controller, last_step : dict):
    possible_actions = deepcopy(ACTIONS)
    microwave = get_object_id(last_step['event'], 'Microwave')[1]
    if microwave['isOpen']:
        # When microwave is open, we cannot toggle it on
        possible_actions['ToggleObject'].remove('Microwave')
    elif microwave['isToggled']:
        # When microwave is toggled on, we cannot open it
        possible_actions['OpenObject'].remove('Microwave')
    if last_step['pickedObject'] == None:
        possible_actions.pop('SliceObject')
        possible_actions.pop('PutObject')
        possible_actions['PickupObject'] *= 2
    else:
        # possible_actions.pop('PickupObject')
        possible_actions['PickupObject'].remove(last_step['pickedObject'])
        possible_actions['MoveObject'] = possible_actions.pop('PickupObject') * 2
        if (not microwave['isOpen'] or not last_step['pickedObject'].startswith('Plate')) and 'Microwave' in possible_actions['PutObject']:
            possible_actions['PutObject'].remove('Microwave')
        if not last_step['pickedObject'].startswith('Knife'):
            possible_actions.pop('SliceObject')
            if last_step['pickedObject'].startswith('Egg') and 'Pan' in possible_actions['PutObject']:
                if last_step['step_number'] < 40:
                    possible_actions['PutObject'].remove('Pan')
                else:
                    possible_actions['PutObject'] += ['Pan'] * 2
            elif 'Pan' in possible_actions['PutObject']:
                possible_actions['PutObject'].remove('Pan')
        else:
            possible_actions['PutObject'] = possible_actions['PutObject'][-1:] * 2
            if last_step['step_number'] < 50:
                possible_actions['SliceObject'].remove('Apple')
            possible_actions['SliceObject'] *= 2
        possible_actions['PutObject'] *= 1
        possible_actions['PutObject'] += possible_actions['PutObject'][-1:] * 2
    possible_actions = [(action_type, object_name) for action_type, object_names in possible_actions.items() for object_name in object_names]
    # Some actions are invalid, so we try to perform an action until we get a valid one
    action_pos = None
    action_counter = 0
    while action_pos is None and action_counter < 20:
        action_type, object_name = possible_actions[np.random.randint(0, len(possible_actions))]
        event, action_pos = perform_action(controller, action_type, object_name, last_step['event'], step_number=last_step['step_number']+1)
        action_counter += 1
    assert action_pos is not None, 'Could not perform any action'
    if last_step['pickedObject'] is not None and \
        last_step['pickedObject'].startswith('Egg') and \
            action_type == 'PutObject' and object_name == 'Pan' and event.metadata['lastActionSuccess']:
        event = controller.step(action='BreakObject',
                                objectId=get_object_id(event, 'Egg')[0])
        ACTIONS['PickupObject'].remove('Egg')
    # Advance physics time step
    event = controller.step(action='Stand')
    event = controller.step(action='Stand')

    pickedObject = None
    if action_type == 'PickupObject':
        pickedObject = object_name
    elif action_type == 'PutObject':
        pickedObject = None
    else:
        pickedObject = last_step['pickedObject']

    new_step = {
        'event': event,
        'pickedObject': pickedObject,
        'action_pos': action_pos,
        'debug_info': {
            'action_type': action_type, 
            'object_name': object_name,
            'pickedObject': pickedObject,
            'action_counter': action_counter
        },
        'step_number': last_step['step_number'] + 1,
    }
    return new_step

def create_targets(debug_info : dict, causal_keys : list):
    targets = np.zeros((len(debug_info['action_type']) + 1, len(causal_keys)), dtype=np.uint8)
    for i in range(len(debug_info['action_type'])):
        object_name = debug_info['object_name'][i]
        action_type = debug_info['action_type'][i]
        if action_type == 'PutObject':
            object_name = debug_info['pickedObject'][i - 1]
        elif action_type == 'NoOp':
            continue
        for j, key in enumerate(causal_keys):
            if key.startswith(object_name):
                if action_type == 'OpenObject' and not key.endswith('_open'):
                    continue
                if action_type == 'ToggleObject' and not key.endswith('_on'):
                    continue
                if action_type in ['PickupObject', 'PutObject'] and \
                    not (key.endswith('_pickedup') or any([key.endswith(f'_center_{pf}') for pf in ['x', 'y', 'z']])):
                    continue
                if action_type == 'SliceObject' and not key.endswith('_sliced'):
                    continue
                targets[i + 1, j] = 1
    return targets

def simplify_latents(latents : np.ndarray, causal_keys : list):
    latents_dict = {key: latents[:, i] for i, key in enumerate(causal_keys)}

    apple_slice_keys = [k for k in causal_keys if k.startswith('Apple_10_Sliced')]
    num_apple_slices = len(set([k.split('_')[3] for k in apple_slice_keys]))
    if num_apple_slices > 0:
        set_sliced = False
        for key in apple_slice_keys:
            if not set_sliced and key.endswith('center_x'):
                latents_dict['Apple_f33eaaa0_sliced'] += (latents_dict[key] != 0.0).astype(np.float32)
                set_sliced = True
            orig_key = 'Apple_f33eaaa0_' + key.split('_', 4)[-1]
            latents_dict[orig_key] += latents_dict[key] * (1 / num_apple_slices)
            latents_dict.pop(key)
    
    egg_cracked_keys = [k for k in causal_keys if k.startswith('Egg_Cracked')]
    set_broken = False
    for key in egg_cracked_keys:
        if not set_broken and key.endswith('center_x'):
            latents_dict['Egg_afaaaca3_broken'] += (latents_dict[key] != 0.0).astype(np.float32)
            set_broken = True
        orig_key = 'Egg_afaaaca3_' + key.split('_', 3)[-1]
        if orig_key not in latents_dict:
            latents_dict[orig_key] = latents_dict[key]
        else:
            latents_dict[orig_key] += latents_dict[key]
        latents_dict.pop(key)
    if 'Egg_afaaaca3_cooked' not in latents_dict:
        latents_dict['Egg_afaaaca3_cooked'] = np.zeros_like(latents_dict['Egg_afaaaca3_center_x'])

    potato_keys = [k for k in causal_keys if k.startswith('Potato')]
    set_sliced = False
    for key in potato_keys:
        if key.endswith('sliced'):
            continue
        elif not set_sliced and key.endswith('center_x') and '_Slice_' in key:
            latents_dict['Potato_e4559da4_sliced'] = (latents_dict[key] != 0.0).astype(np.float32)
            set_sliced = True
        latents_dict.pop(key)
    
    plate_keys = [k for k in causal_keys if k.startswith('Plate')]
    for t in range(1, latents.shape[0]):
        if latents_dict[plate_keys[0]][t] == 0.0: # center x being zero - in Microwave
            for k in plate_keys:
                latents_dict[k][t] = latents_dict[k][t-1]
    
    causal_keys = sorted(list(latents_dict.keys()))
    latents = np.stack([latents_dict[key] for key in causal_keys], axis=1)
    return latents, causal_keys

def downscale_images(images : np.ndarray):
    images = torch.from_numpy(images)
    images = images.permute(0, 3, 1, 2)
    images = F.interpolate(images.float(), 
                           scale_factor=(0.5, 0.5),
                           mode='bilinear')
    images = images.permute(0, 2, 3, 1)
    images = images.numpy()
    # To numpy uint8
    images = images.astype(np.uint8)
    return images

def generate_sequence(seed : int, output_folder : str, num_frames : int = 200, prefix : str = "", save_segmentations : bool = False):
    print('-> Seed:', seed)
    attempts = 0
    success = False
    while not success and attempts < 10:
        try:
            controller, event = initialize_environment(seed)
            success = True
        except:
            attempts += 1
            continue
    assert success, 'Failed to initialize environment after 10 attempts'
    last_step = {'event': event, 'pickedObject': None, 'step_number': 0}
    collected_frames = np.zeros((num_frames, RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    collected_actions = np.zeros((num_frames, 2), dtype=np.float32)
    collected_states = defaultdict(lambda : np.zeros((num_frames,), dtype=np.float32))
    collected_segmentations = dict()
    debug_info = defaultdict(lambda : [])
    collected_frames[0] = event.frame
    for key, val in get_environment_state(event).items():
        collected_states[key][0] = val
    if save_segmentations:
        for key, val in get_object_segmentations(event).items():
            collected_segmentations[key] = np.zeros((num_frames, *val.shape), dtype=val.dtype)
            collected_segmentations[key][0] = val
    for i in range(1,num_frames):
        last_step = perform_random_action(controller, last_step)
        collected_frames[i] = last_step['event'].frame
        collected_actions[i] = last_step['action_pos']
        for key, val in get_environment_state(last_step['event']).items():
            collected_states[key][i] = val
        for key, val in last_step['debug_info'].items():
            debug_info[key].append(val)
        if save_segmentations:
            for key, val in get_object_segmentations(last_step['event']).items():
                collected_segmentations[key][i] = val

    controller.stop()
    collected_frames = downscale_images(collected_frames)
    causal_keys = sorted(list(collected_states.keys()))
    latents = np.stack([collected_states[key] for key in causal_keys], axis=1)
    latents, causal_keys = simplify_latents(latents, causal_keys)
    targets = create_targets(debug_info, causal_keys)
    np.savez_compressed(os.path.join(output_folder, f'{prefix}seq_{str(seed).zfill(6)}.npz'), 
                        frames=collected_frames.transpose(0, 3, 1, 2), 
                        actions=collected_actions, 
                        latents=latents, 
                        targets=targets,
                        causal_keys=causal_keys,
                        **{'segm_'+k: collected_segmentations[k] for k in collected_segmentations})
    debug_info['causal_keys'] = causal_keys
    debug_info['seed'] = int(seed)
    with open(os.path.join(output_folder, f'{prefix}seq_{str(seed).zfill(6)}_infos.json'), 'w') as f:
        json.dump(debug_info, f, indent=4)
    return collected_frames, collected_actions, collected_states

def print_time(time : float):
    string = ''
    if time > 3600:
        string += f'{int(time // 3600)}h '
        time = time % 3600
    if time > 60:
        string += f'{int(time // 60)}m '
        time = time % 60
    string += f'{int(time)}s'
    return string

if __name__ == '__main__':
    output_folder = 'data/ithor/'
    num_frames = 100
    num_sequences = 1
    prefix = 'train'
    overwrite = True
    save_segmentations = False
    output_folder = os.path.join(output_folder, prefix)
    hash = sha256(prefix.encode())
    offset_seed = np.frombuffer(hash.digest(), dtype='uint32')[0]

    orig_mov_position = MOVABLE_POSITION_DICT
    orig_counter_positions = COUNTER_POSITIONS
    orig_actions = ACTIONS
    orig_pickup = PICKUP
    os.makedirs(output_folder, exist_ok=True)
    start_time = time.time()
    for seq_idx in range(1, 1+num_sequences):
        MOVABLE_POSITION_DICT = deepcopy(orig_mov_position)
        COUNTER_POSITIONS = deepcopy(orig_counter_positions)
        ACTIONS = deepcopy(orig_actions)
        PICKUP = deepcopy(orig_pickup)
        out_file = os.path.join(output_folder, f'{prefix}_seq_{str(offset_seed + seq_idx).zfill(6)}.npz')
        if os.path.exists(out_file) and not overwrite:
            continue
        print('#' * 50 + '\n' + f'Generating sequence {seq_idx} of {num_sequences}...' + '\n' + '#' * 50)
        collected_frames, collected_actions, collected_states = generate_sequence(seed=offset_seed + seq_idx, 
                                                                                  output_folder=output_folder, 
                                                                                  num_frames=num_frames,
                                                                                  save_segmentations=save_segmentations,
                                                                                  prefix=f'{prefix}_')
        if seq_idx == 1 and prefix == 'train':
            collected_frames = [frame for frame in collected_frames]
            exmp_folder = os.path.join(output_folder, f'{prefix}_seq_{str(seq_idx).zfill(6)}')
            os.makedirs(exmp_folder, exist_ok=True)
            imageio.mimsave(os.path.join(exmp_folder, f'all_imgs.mov'), 
                            collected_frames, fps=2)
            for i, frame in enumerate(collected_frames):
                imageio.imwrite(os.path.join(exmp_folder,
                                             f'img_{str(i).zfill(6)}.png'), frame)
        else:
            del collected_frames, collected_actions, collected_states
        print('-' * 50 + '\n' + f'Finished sequence {seq_idx} of {num_sequences}...' + '\n' + '-' * 50)
        print(f'Elapsed time: {print_time(time.time() - start_time)}')
        if seq_idx != num_sequences:
            print(f'Estimated time remaining: {print_time((time.time() - start_time) * (num_sequences - seq_idx) / seq_idx)}')