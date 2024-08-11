#!/bin/bash
nohup python main.py --base configs/VQGAN_blue.yaml -t True --gpus 1 > output.log 2>&1 &
