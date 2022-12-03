# Baseline tracking

## PART1 Data preparation
- Detection result and frames meta data from [here](https://drive.google.com/drive/folders/13jmwcS2qu89QftSmrWmGpgQu20gF8YPl?usp=share_link).
- If you want to evaluate and visualize, you will need to download nuscenes trainval dataset from [their website](https://www.nuscenes.org/nuscenes#download).

## PART2 Environment setup
1. `git clone https://github.com/derekray311511/nusc_baseline_tracking.git`  
2. cd to the workspace directory you just clone.  
    `cd nusc_baseline_tracking`  
3. `cd docker && docker build . -t tracking`
4. run the docker image and create a container  
    `bash run.sh`  
    run the same container in other terminal  
    `docker exec -it tracking bash`  
5. Create a virtual path to data directory in docker
    `cd /home/Student/Tracking`  
    `ln -fsv /data data`   
6. Put the `detection_result.josn` and `frames_meta.json` (download from PART1) into your data directory

## PART3 Tracking


## PART4 Visualization
Need to download dataset from nuscenes trainval dataset from [their website](https://www.nuscenes.org/nuscenes#download).
1. `cd ros_ws`
2. `catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3`
3. `source devel/setup.bash`
4. First terminal: `roscore`
5. Second terminal: `rviz -d configs/track.rviz` (can work outside of the docker container)
6. Third terminal: `bash src/visualize.sh`