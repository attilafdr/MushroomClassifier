#!/bin/sh

torch-model-archiver \
  --model-name mushnet18 \
  --version 0.1 \
  --model-file src/mushroom/models/resnet18.py \
  --serialized-file model-store/ckp0029.pt \
  --handler image_classifier