#!/bin/bash

for((i=0; i<1; i++)); do {
	python3 main.py  --gpu 0  --task_config  'configs/color_config.yaml' --model_config 'configs/model_config.yaml'  --save_dir './saved_results'
  echo "DONE!"
} & done
wait

