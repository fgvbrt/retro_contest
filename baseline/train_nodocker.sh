#!/usr/bin/env bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="4" python ppo2_agent.py --config config.yaml config_train.yaml
