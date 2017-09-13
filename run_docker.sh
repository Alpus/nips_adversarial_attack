#!/usr/bin/env bash


nvidia-docker run \
    -v /home/alpus/Work/kaggle/attack/sample/input_dir:/input_images \
    -v /home/alpus/Work/kaggle/attack/sample/output_dir:/output_images \
    -v /home/alpus/Work/kaggle/attack/sample/submit:/code \
    -w /code \
    alpus/tensorflow_attack:gpu \
    ./run_attack.sh \
    /input_images \
    /output_images \
    15
