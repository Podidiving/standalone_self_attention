#!/usr/bin/env bash

# classification problem
PYTHONPATH=. python3.6 ./trainer/main.py --configs ./configs/casia_classification_small/classification_config_general.yml ./configs/casia_classification_small/classification_config_stage_1.yml --type classification
PYTHONPATH=. python3.6 ./trainer/main.py --configs ./configs/casia_classification_small/classification_config_general.yml ./configs/casia_classification_small/classification_config_stage_2.yml --type classification
