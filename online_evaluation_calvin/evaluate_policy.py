"""Modified from
https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py
"""
import os
import gc
from typing import Tuple, Optional, List
import random
import logging
from pathlib import Path

import tap
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import yaml
from tqdm import tqdm

from utils.common_utils import get_gripper_loc_bounds
from online_evaluation_calvin.evaluate_model import create_model
from online_evaluation_calvin.evaluate_utils import (
    prepare_visual_states,
    prepare_proprio_states,
    count_success,
    get_env_state_for_initial_condition,
    collect_results,
    write_results,
    get_log_dir
)
from online_evaluation_calvin.multistep_sequences import get_sequences
from online_evaluation_calvin.evaluate_utils import get_env

logger = logging.getLogger(__name__)

EP_LEN = 60
NUM_SEQUENCES = 1000
EXECUTE_LEN = 20


class Arguments(tap.Tap):
    # Online enviornment
    calvin_dataset_path: Path = "/home/tsungwek/repos/calvin/dataset/task_ABC_D"
    calvin_model_path: Path = "/home/tsungwek/repos/calvin/calvin_models"
    calvin_demo_tasks: Optional[List[str]] = None
    device: str = "cuda"
    text_encoder: str = "clip"
    text_max_length: int = 16
    save_video: int = 0

    # Offline data loader
    seed: int = 0
    tasks: Tuple[str, ...] # indicates the environment
    checkpoint: Path
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    calvin_gripper_loc_bounds: Optional[str] = None
    relative_action: int = 0

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "eval_logs" / "calvin"

    # Model
    action_dim: int = 7 # dummy, as DiffuserActor assumes action_dim is 7
    image_size: str = "256,256" # decides the FPN architecture
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = 'quat'
    quaternion_format: str = 'wxyz'
    diffusion_timesteps: int = 100
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 3
    num_history: int = 0
    interpolation_length: int = 2 # the number of steps to reach keypose


def make_env(dataset_path, show_gui=True, split="validation", scene=None):
    val_folder = Path(dataset_path) / f"{split}"
    if scene is not None:
        env = get_env(val_folder, show_gui=show_gui, scene=scene)
    else:
        env = get_env(val_folder, show_gui=show_gui)

    return env


def evaluate_policy(model, env, conf_dir, eval_log_dir=None, save_video=False,
                    sequence_indices=[]):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: an instance of CalvinBaseModel
        env: an instance of CALVIN_ENV
        conf_dir: Path to the directory containing the config files of CALVIN
        eval_log_dir: Path where to log evaluation results
        save_video: a boolean indicates whether to save the video
        sequence_indices: a list of integers indicates the indices of the
            instruction chains to evaluate

    Returns:
        results: a list of integers indicates the number of tasks completed
    """
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results, tested_sequence_indices = collect_results(eval_log_dir)

    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if sequence_indices and seq_ind not in sequence_indices:
            continue
        if seq_ind in tested_sequence_indices:
            continue
        result, videos = evaluate_sequence(
            env, model, task_oracle, initial_state,
            eval_sequence, val_annotations, save_video
        )
        write_results(eval_log_dir, seq_ind, result, eval_sequence[0], val_annotations[eval_sequence[0]])
        results.append(result)
        str_results = (
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
            for i, v in enumerate(count_success(results))]) + "|"
        )
        print(str_results + "\n")

        if save_video:
            import moviepy.video.io.ImageSequenceClip
            from moviepy.editor import vfx
            clip = []
            import cv2
            for task_ind, (subtask, video) in enumerate(zip(eval_sequence, videos)):
                for img_ind, img in enumerate(video):
                    cv2.putText(img,
                                f'{task_ind}: {subtask}',
                                (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,
                                (0, 0, 0),
                                1,
                                2)
                    video[img_ind] = img
                clip.extend(video)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(clip, fps=30)
            clip.write_videofile(f"calvin_seq{seq_ind}.mp4")


    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence,
                      val_annotations, save_video):
    """
    Evaluates a sequence of language instructions.

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_checker: an indicator of whether the current task is completed
        initial_state: a tuple of `robot_obs` and `scene_obs`
            see: https://github.com/mees/calvin/blob/main/dataset/README.md#state-observation
        eval_sequence: a list indicates the instruction chain
        val_annotations: a dictionary of task instructions
        save_video: a boolean indicates whether to save the video

    Returns:
        success_counter: an integer indicates the number of tasks completed
        video_aggregator: a list of lists of images that shows the trajectory
            of the robot

    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter, video_aggregators = 0, []
    for subtask in eval_sequence:
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        # if not subtask == 'push_pink_block_right':
        #     break

        '''
        push_pink_block_right
        push_blue_block_right
        push_red_block_right
        '''

        # use_inpainting = (success_counter==0)
        use_inpainting = True
        success, video = rollout(env, model, task_checker,
                                 subtask, lang_annotation, use_inpainting= use_inpainting)
        video_aggregators.append(video)

        if success:
            success_counter += 1
        else:
            return success_counter, video_aggregators
        
        break
    return success_counter, video_aggregators


def rollout(env, model, task_oracle, subtask, lang_annotation, use_inpainting=False):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_oracle: an indicator of whether the current task is completed
        subtask: a string indicates the task name
        lang_annotation: a string indicates the instruction of the task

    Returns:
        Success/Fail: a boolean indicates whether the task is completed
        video: a list of images that shows the trajectory of the robot
    """
    video = [] # show video for debugging
    obs = env.get_obs()
    start_obs = obs.copy()

    model.reset()
    start_info = env.get_info()

    # print('------------------------------')
    # print(f'task: {lang_annotation}')
    # video.append(obs["rgb_obs"]["rgb_static"])

    # if use_inpainting:

    #     obj_list =['red', 'blue', 'pink', 'sliding', 'drawer','button', 'turn on the light bulb']
    #     obj_poses = {
    #         'red': obs["scene_obs"][6:9],
    #         'blue': obs["scene_obs"][12:15],
    #         'pink': obs["scene_obs"][18:21],
    #         'sliding': np.array([-0.08679207, -0.0115027 ,  0.54333853]),
    #         'drawer': np.array([0.1826129 , -0.31640463,  0.374497]),
    #         'turn on the light bulb': np.array([0.24155065, 0.03409243, 0.55943577]),
    #         'button': np.array([-0.10994109, -0.12844143,  0.48871266]),
    #         }
        
    #     '''
    #     turn on the lightbulb / target pose
    #     [ 0.2917001 , -0.0173986 ,  0.54783777]

    #     turn off lightbulb / target pose
    #     [0.24155065, 0.03409243, 0.55943577]
    #     '''
    #     target_obj = None
    #     target_obj_list = []
    #     for obj in obj_list:
    #         if obj in lang_annotation:
    #             target_obj_list.append(obj)
                
    #     if len(target_obj_list) == 1:
    #         target_obj = target_obj_list[0]
    #     elif len(target_obj_list) > 1:
    #         print('!!! multiple target objects:', target_obj_list)
        
    #     if 'grasped' in lang_annotation:
    #         target_obj =  None
    #         print('!!! grasped object')

    #     if 'block' in lang_annotation and target_obj in ['drawer', 'sliding']:
    #         target_obj = None
    #         print('!!! block object to container')
            
    #     if target_obj is not None:
    #         print('!!! target object:', target_obj)
    #     else:
    #         print('!!! no target object')

    #     pre_leading = True

    pbar = tqdm(range(EP_LEN))
    for step in pbar:
        obs = prepare_visual_states(obs, env)
        obs = prepare_proprio_states(obs, env)
        lang_embeddings = model.encode_instruction(lang_annotation, model.args.device)
        with torch.cuda.amp.autocast():
            trajectory = model.step(obs, lang_embeddings)

        for act_ind in range(min(trajectory.shape[1], EXECUTE_LEN)):
            # calvin_env executes absolute action in the format of:
            # [[x, y, z], [euler_x, euler_y, euler_z], [open]]
            curr_action = [
                trajectory[0, act_ind, :3],
                trajectory[0, act_ind, 3:6],
                trajectory[0, act_ind, [6]]
            ]

            # if use_inpainting:
            #     # change as the object is:
            #     if pre_leading and target_obj is not None:
                    
            #         to_obj_dist = np.linalg.norm(curr_action[0] - obj_poses[target_obj])
            #         if to_obj_dist < 0.1:
            #             pre_leading = False
            #             print('the object is leading now')
            #         else:
            #             ratio = 0.1
            #             curr_action[0] = ratio * curr_action[0] + (1 - ratio) * obj_poses[target_obj]


            pbar.set_description(f"step: {step}")
            curr_proprio = obs['proprio']
            obs, _, _, current_info = env.step(curr_action)
            obs['proprio'] = curr_proprio

            # check if scene in start info is different from current info.
            change = start_obs["scene_obs"] - obs["scene_obs"]
            change_diff = np.linalg.norm(change)

            no_change = change_diff < 0.1
            if not no_change:
                # print('the state changed now', np.where(np.abs(change)>0.01))
                a = 1




            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )

            video.append(obs["rgb_obs"]["rgb_static"])

            if len(current_task_info) > 0:
                print('*** task successful***')
                return True, video

    return False, video


def get_calvin_gripper_loc_bounds(args):
    with open(args.calvin_gripper_loc_bounds, "r") as stream:
       bounds = yaml.safe_load(stream)
       min_bound = bounds['act_min_bound'][:3]
       max_bound = bounds['act_max_bound'][:3]
       gripper_loc_bounds = np.stack([min_bound, max_bound])

    return gripper_loc_bounds


def main(args):

    # These location bounds are extracted from language-annotated episodes
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )

    # These location bounds are extracted from every episode in play trajectory
    if args.calvin_gripper_loc_bounds is not None:
        args.calvin_gripper_loc_bounds = get_calvin_gripper_loc_bounds(args)

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # evaluate a custom model
    model = create_model(args)

    # sequence_indices = [
    #     i for i in range(args.local_rank, NUM_SEQUENCES, int(os.environ["WORLD_SIZE"]))
    # ]

    sequence_indices = [
        i for i in range(0, NUM_SEQUENCES, 1)
    ]

    env = make_env(args.calvin_dataset_path, show_gui=False)
    evaluate_policy(model, env,
                    conf_dir=Path(args.calvin_model_path) / "conf",
                    eval_log_dir=args.base_log_dir,
                    sequence_indices=sequence_indices,
                    save_video=args.save_video)

    results, sequence_inds = collect_results(args.base_log_dir)
    str_results = (
        " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
        for i, v in enumerate(count_success(results))]) + "|"
    )
    print(f'Load {len(results)}/1000 episodes...')
    print(str_results + "\n")

    del env
    gc.collect()

if __name__ == "__main__":
    args = Arguments().parse_args()
    # args.local_rank = int(os.environ["LOCAL_RANK"])

    # DDP initialization
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    main(args)
