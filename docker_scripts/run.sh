#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")"; cd ..; pwd)"
STARFCPY_ROOT="/opt/STAR-FC"


usage() {
    echo "Usage: $0 [-v] -c <config_file_path>"
    echo "Options:"
    echo "-v \t visualization on"
    echo "-c \t path to config file with extension .ini (see config_files for examples)"
}

vis_flag=''
config_file_path=''

while getopts "h?vc:" opt; do
    case "$opt" in
        h|\?)
            usage
            exit 0
            ;;
        v)  vis_flag='-v'
            ;;
        c)  config_file_path=$OPTARG
            ;;
        esac
done
shift "$((OPTIND-1))"

if [ -z "$config_file_path" ]; then
    echo "ERROR: config file not provided!"
    usage
    exit 1
fi

xhost +local:starfcpy
nvidia-docker run -it \
  --name starfcpy \
  -h starfcpy \
  -v ${PROJECT_ROOT}:${STARFCPY_ROOT} \
  -v /dev/input \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -w ${STARFCPY_ROOT} \
  --rm \
  starfcpy python3 src/STAR_FC.py $vis_flag -c $config_file_path
xhost -local:starfcpy
