#!/usr/bin/env bash

# classification problem
PYTHONPATH=. python3.6 ./trainer/main.py --config ./configs/classification_config_casia.yml --type classification
