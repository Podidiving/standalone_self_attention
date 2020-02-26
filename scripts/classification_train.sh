#!/usr/bin/env bash

# classification problem
PYTHONPATH=. python3.6 ./trainer/main.py --config ./configs/classification_config.yml --type classification
