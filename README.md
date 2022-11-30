# Baseline tracking

## Data preparation
- Detection result and frames meta data from [here](https://drive.google.com/drive/folders/13jmwcS2qu89QftSmrWmGpgQu20gF8YPl?usp=share_link).
- If you want to evaluate by yourself, you will need to download nuscenes trainval dataset from [their website](https://www.nuscenes.org/nuscenes#download).

## Environment setup
1. `git clone https://github.com/derekray311511/nusc_baseline_tracking.git`
2. cd to the workspace directory you just clone.
    `cd nusc_baseline_tracking`
3. `cd docker && docker build . -t tracking`
4. run the docker image and create a container
    `bash run.sh`
    run the same container in other terminal
    `docker exec -it tracking bash`
5. Create a virtual path to data directory
    `cd /home/Student/Tracking`
    `ln -fsv /data data` 
6. Put the `detection_result.josn` and `frames_meta.json` (download from ???) into your data directory