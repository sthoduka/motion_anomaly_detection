This repository contains the code for the paper "Using Visual Anomaly Detection for Task Execution Monitoring"

## Dependencies:
See [requirements.txt](requirements.txt).

The main ones are:
* torch=2.3.0
* pytorch-lightning=2.2.5

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

## Citation
Please cite this work in your publications if you found it useful. Here is the BibTeX entry:

```
@inproceedings{thoduka2021using,
  title={{Using Visual Anomaly Detection for Task Execution Monitoring}},
  author={Thoduka, Santosh and Gall, Juergen and Pl{\"o}ger, Paul G},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4604--4610},
  year={2021},
  organization={IEEE}
}
```
