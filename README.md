# Online Semantic Scene Representation Using Neural Fields
   In the field of robotics, strong perception systems that capture visual and meaningful information about scenes would prove to be useful for many downstream tasks. Furthermore, these scene representations must be generated online in a matter of minutes, as opposed to hours or days. For our final project, we present Online Semantic Scene Representations Using Neural Fields, a method for generating semantic and photometric scene representations that leverages multi-resolution hash encodings to significantly speed up training over previous methods.
## 
- Report on implementation details with qualitative results: [Online Semantic Segmentation Using Neural Fields](report.pdf)
## Pre-requisites
- python 3.8.16
- Bosdyn SDK https://dev.bostondynamics.com/docs/python/quickstart#pip-installation
- CLIP https://github.com/openai/CLIP
- Segment Anything https://github.com/facebookresearch/segment-anything
- Torch Ngp pre-requisites https://github.com/ashawkey/torch-ngp

## Running the robot
- This script will run a data capture script using the Spot robot to get images from its gripper camera.
./run_robot.sh

## Running Segmentation Code
- This script will run our automated semantic segmentation pipeline using CLIP and SAM
./run_segmentation.sh

## Running Colmap
- This script will run colmap in order to get the camera poses for the images taken by spot
./run_colmap.sh

## Setup
- Once all of the above scripts are run, you will need to manually move "images", "segments", and "transforms.json" to a folder in semantic-torch-ngp/data/nerf/

## Running Semantic Torch Ngp Code
- This script will run the Semantic NeRF model inside a gui. 
./run_nerf_gui.sh
