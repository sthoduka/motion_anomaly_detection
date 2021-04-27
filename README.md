This repository contains the code for the paper "Using Visual Anomaly Detection for Task Execution Monitoring"

## Dependencies:
See [requirements.txt](requirements.txt).

The main ones are:
* torch=1.6.0
* pytorch-lightning=0.9.0

## Clone
    git clone --recursive https://github.com/sthoduka/motion_anomaly_detection.git

## Generate optical flow images
Follow the instructions [here](apps/optical_flow).

## Train
    python main.py \
      --video_root=<path to training data folder> \
      --val_video_root=<path to validation data folder> \
      --test_video_root=<path to test data folder> \
      --sample_size=64 \
      --batch_size=128 \
      --default_root_dir=<path to tensorboard logs folder> \
      --row_log_interval=10 \
      --learning_rate=0.0001 \
      --max_epochs=50 \
      --gpus=1 \
      --flow_type=normal_masked \
      --prediction_offset_start=5 \
      --prediction_offset=9

## Compute expected and observed camera motion
Follow the instructions [here](apps/camera_motion).

## Generate rendered robot body images
The dataset already includes the rendered robot body images. If you want to regenerate them or render them for your own dataset/robot, follow the instructions [here](apps/robot_render).

