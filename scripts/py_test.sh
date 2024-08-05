main_dir=Planner_Calvin

dataset=./data/calvin/packaged_ABC_D/training
valset=./data/calvin/packaged_ABC_D/validation

lr=3e-4
wd=5e-3
dense_interpolation=1
interpolation_length=20
num_history=3
diffusion_timesteps=25
B=30
C=192
ngpus=1
backbone=clip
image_size="256,256"
relative_action=1
fps_subsampling_factor=3
lang_enhanced=1
gripper_loc_bounds=tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
gripper_buffer=0.01
val_freq=5000
quaternion_format=wxyz

export PYTHONPATH=`pwd`:$PYTHONPATH

python  online_evaluation_calvin/evaluate_policy.py \
    --calvin_dataset_path calvin/dataset/task_ABC_D \
    --calvin_model_path calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks A B C D\
    --backbone clip \
    --gripper_loc_bounds tasks/calvin_rel_traj_location_bounds_task_ABC_D.json \
    --gripper_loc_bounds_buffer 0.01 \
    --calvin_gripper_loc_bounds calvin/dataset/task_ABC_D/validation/statistics.yaml \
    --embedding_dim 192 \
    --action_dim 7 \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps 25 \
    --interpolation_length 20 \
    --num_history 3 \
    --relative_action 1 \
    --fps_subsampling_factor 3 \
    --lang_enhanced 1 \
    --save_video 0 \
    --base_log_dir train_logs/Planner_Calvin/pretrained/eval_logs/ \
    --quaternion_format wxyz \
    --checkpoint train_logs/diffuser_actor_calvin.pth
