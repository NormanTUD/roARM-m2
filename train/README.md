# What is this repo?

This offers python scripts to control the Waveshare RoArm-M2 via USB. This is all just a beta.

## What is a "policy"?

A policy is an AI model trained on robot movement data. It first needs to be recorded, then trained, and can then be used. It assumes you have a camera enabled.

## What do the files do?

- `record_policy.py`: Start here to record a new policy
- `train_policy.py`: Train a model on a set of records
- `run_policy.py `: Run a trained model
- `roarm_m2s.py`: Abstraction layer for the hardware of the RoArm-M2
