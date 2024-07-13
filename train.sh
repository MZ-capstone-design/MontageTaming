#!/bin/bash
nohup python main.py --base configs/VQGAN_blue.yaml -t True --gpus 0 > output.log 2>&1 &
