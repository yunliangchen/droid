#!/bin/bash

xhost +local:root



# SCRIPT_PATH="/home/lawchen/project/robosuite_robomimic/docker/examples"
# OUTPUT_PATH="/home/lawchen/project/robosuite_robomimic"

# docker run \
#   -it \
#   --gpus all \
#   --net host \
#   -e DISPLAY \
#   -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -v $(pwd)/robosuite:/robosuite \
#   -v $(pwd)/robomimic:/robomimic \
#   --rm \
#   -v $SCRIPT_PATH:/scripts \
#   -v $OUTPUT_PATH:/output \
#   my_robosuite_robomimic_image \
#   /bin/bash -c "/scripts/script.sh > /output/output.txt"



docker run \
  -it \
  --gpus all \
  --net host \
  -e DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home:/home \
  --rm \
  lawchen_droid_image:1.0 \
  /bin/bash #-c "cd /robomimic/robomimic/scripts && python3 dataset_states_to_obs.py --dataset  /robomimic/datasets/lift/ph/demo_v141.hdf5 --output_name test.hdf5 --done_mode 2 --camera_names agentview agentview_1 --camera_height 84 --camera_width 84"

xhost -local:root
