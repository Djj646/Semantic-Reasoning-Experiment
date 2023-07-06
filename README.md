# Semantic-Reasoning-Experiment

the experiment part of the rcar 2023 'Knowledge Distillation on Driving Intention Generator: Learn Human-like Semantic Reasoning'.

## Model Download

[baidu netdisk](https://pan.baidu.com/s/1Pq90IUN3fEwoNRVCq0auOw?pwd=2022) 

## How To Use

### CollectSimData

Collect data from carla simulator, support manual driving mode:

```shell
python main.py
```

### KITTIDatasetProcessing

Process kitti dataset to get the nav map corresponding to the position:

```shell
python pose.py
```

### NPYViewer

View the `.npy` format point cloud data collected by lidar:

```shell
python NPYViewer.py
```

### SimDataProcessingPM

Process data collected from the CARLA simulatior:

- A certain amount of time stamps:
  - generate real pm: `pm.py`
  - Feed data into the model to generate trajectories : `img2pm.py`
  - Feed data into the model to generate trajectories with multi-weathers or fake-nav: `img2pm_weather.py`, `img2pm_fakenav.py`

- Single time stamp: `*_single.py`

Support to generate mp4 video for inspection.

### Train

the folder 'Train' contains the definition scripts of the model.

### TrajectoryEvaluation

The evaluation of the trajectory includes three indicators, namely IoU, cover rate and yaw angle change. Its core implementation is in `./utils/evaluation.py`.

*Before using the evaluation script, you need to pre-generate all the trajectories.

- A certain amount of time stamps:
  - specified interval: `eval_interval.py`
  - only turning: `eval_turning.py` and add the running parameter `--turning`
  - only straight: `eval_turning.py` and add the running parameter `--st`
- Single time stamp: `eval_single.py`

Here are some instrumental scripts:

- `monitor.py`: monitor which turns and their corresponding time stamps in the entire trip.
- `total_average.py`: calculates the average of the turn evaluation and the straight-ahead evaluation
