#!/usr/bin/env bash


PYTHONPATH=. python3.6 ./trainer/main.py --configs \
./configs/casia_classification_0/classification_config_general.yml \
./configs/casia_classification_small/classification_config_stage_1.yml \
--type classification
